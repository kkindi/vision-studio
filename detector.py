"""
detector.py - YOLO detection engine with automatic hardware optimization.

This module is the "brain" of Vision Studio. It:
  * detects the host hardware (CUDA GPU / Apple MPS / CPU) and picks the best
    model size + input resolution automatically,
  * runs detection on single images,
  * runs detection + multi-object tracking on video / camera frames so we can
    count UNIQUE objects (e.g. how many distinct cars passed), not just how many
    are visible in the current frame.

Everything runs fully offline once the model weights exist in ./models.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(HERE, "models")

# Tuned tracker config (keeps stable IDs on low-FPS CPU so objects aren't
# re-counted). Falls back to Ultralytics' built-in if the file is missing.
_TRACKER_CFG = os.path.join(HERE, "trackers", "bytetrack_robust.yaml")
TRACKER = _TRACKER_CFG if os.path.isfile(_TRACKER_CFG) else "bytetrack.yaml"

# ---------------------------------------------------------------------------
# Model family. YOLO26 is the newest released Ultralytics model (NMS-free,
# faster on CPU than YOLO11). To move to a newer family later, change this one
# line - e.g. "yolo27" once its weights are published.
# ---------------------------------------------------------------------------
MODEL_FAMILY = "yolo26"
# Older families used only if the current one cannot be downloaded (offline)
# but an older weight file is already sitting in ./models.
FALLBACK_FAMILIES = ("yolo11",)

_SIZE_SUFFIX = {"nano": "n", "small": "s", "medium": "m", "large": "l"}

# Friendly names shown in the UI -> weight file used by Ultralytics.
MODEL_FILES = {key: f"{MODEL_FAMILY}{sfx}.pt" for key, sfx in _SIZE_SUFFIX.items()}


def resolve_weights(name_or_path: str) -> str:
    """Map a weight name to a path inside ./models.

    If the file is not there yet Ultralytics downloads it to exactly that path
    on first use (internet needed once), so every model ends up in ./models
    and the app is offline-ready afterwards."""
    if os.path.isfile(name_or_path):
        return name_or_path
    if os.path.dirname(name_or_path):          # explicit path that is missing
        return name_or_path
    return os.path.join(MODELS_DIR, name_or_path)


def _local_fallback(size_key: str) -> str | None:
    """An already-downloaded older-family weight of the same size, if any."""
    sfx = _SIZE_SUFFIX.get(size_key)
    if not sfx:
        return None
    for family in FALLBACK_FAMILIES:
        cand = os.path.join(MODELS_DIR, f"{family}{sfx}.pt")
        if os.path.isfile(cand):
            return cand
    return None


@dataclass
class HardwareProfile:
    """The result of probing the machine and deciding how to run."""
    device: str            # 'cuda', 'mps', or 'cpu'
    device_name: str       # human readable
    total_ram_gb: float
    gpu_vram_gb: float
    recommended_model: str  # one of MODEL_FILES keys
    imgsz: int             # inference resolution
    half: bool             # fp16 (GPU only)
    summary: str           # one-line description for the UI

    @staticmethod
    def detect() -> "HardwareProfile":
        import torch

        try:
            import psutil
            ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 1)
            cores = psutil.cpu_count(logical=False) or os.cpu_count() or 4
        except Exception:
            ram_gb = 8.0
            cores = os.cpu_count() or 4

        # --- NVIDIA / CUDA -----------------------------------------------
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            vram = round(props.total_memory / (1024 ** 3), 1)
            if vram >= 12:
                model = "large"
            elif vram >= 8:
                model = "medium"
            elif vram >= 4:
                model = "small"
            else:
                model = "nano"
            return HardwareProfile(
                device="cuda", device_name=props.name, total_ram_gb=ram_gb,
                gpu_vram_gb=vram, recommended_model=model, imgsz=640, half=True,
                summary=f"GPU: {props.name} ({vram} GB) - {model} @ 640px, fp16",
            )

        # --- Apple Silicon / MPS (macOS only) ----------------------------
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return HardwareProfile(
                device="mps", device_name="Apple GPU (MPS)", total_ram_gb=ram_gb,
                gpu_vram_gb=0.0, recommended_model="small", imgsz=640, half=False,
                summary="Apple GPU (MPS) - small @ 640px",
            )

        # --- CPU fallback (tuned by core count / RAM) --------------------
        # YOLO26-nano runs ~25 FPS at 640px on an 8-core laptop CPU, so strong
        # CPUs get the full native resolution; weaker ones trade a little
        # accuracy for speed.
        if cores >= 8 and ram_gb >= 15:      # "16 GB" machines report ~15.8
            model, imgsz = "nano", 640
        elif cores >= 4:
            model, imgsz = "nano", 512
        else:
            model, imgsz = "nano", 416
        cpu_name = _cpu_name()
        return HardwareProfile(
            device="cpu", device_name=cpu_name, total_ram_gb=ram_gb,
            gpu_vram_gb=0.0, recommended_model=model, imgsz=imgsz, half=False,
            summary=f"CPU: {cpu_name} ({cores} cores) - {model} @ {imgsz}px",
        )


def _cpu_name() -> str:
    try:
        from ultralytics.utils.torch_utils import get_cpu_info
        name = get_cpu_info()
        if name:
            return name
    except Exception:
        pass
    try:
        import platform
        return platform.processor() or "CPU"
    except Exception:
        return "CPU"


def _precision_kwargs(fp16: bool) -> dict:
    """Half-precision (fp16) inference argument for the installed Ultralytics.

    Newer releases renamed ``half=True`` to ``quantize=16`` and warn on the
    old name; CPU runs need no argument at all (always fp32)."""
    if not fp16:
        return {}
    try:
        from ultralytics.cfg import DEFAULT_CFG_DICT
        if "quantize" in DEFAULT_CFG_DICT:
            return {"quantize": 16}
    except Exception:
        pass
    return {"half": True}


class Detector:
    """Wraps a YOLO model for image prediction and video/camera tracking."""

    def __init__(self, model_key_or_path: str, profile: HardwareProfile):
        from ultralytics import YOLO

        self.note = ""     # set when we had to fall back to an older model
        if model_key_or_path in MODEL_FILES:
            weights = resolve_weights(MODEL_FILES[model_key_or_path])
            try:
                self.model = YOLO(weights)
            except Exception as exc:
                # Usually: first use of this size while offline. Use an older
                # family that is already on disk rather than failing.
                fallback = _local_fallback(model_key_or_path)
                if not fallback:
                    raise RuntimeError(
                        f"Could not load or download {os.path.basename(weights)} "
                        f"(internet is needed the first time a model size is "
                        f"used).\n{exc}") from exc
                self.note = (f"{os.path.basename(weights)} unavailable "
                             f"(offline?) - using {os.path.basename(fallback)}")
                weights = fallback
                self.model = YOLO(weights)
        else:
            weights = resolve_weights(model_key_or_path)
            self.model = YOLO(weights)

        self.profile = profile
        self.device = profile.device
        self.imgsz = profile.imgsz
        self.half = profile.half
        self._precision = _precision_kwargs(profile.half)
        self.names = self.model.names  # {id: "person", ...}
        self.weights = weights

    # -- single image -----------------------------------------------------
    def predict_image(self, frame_bgr, conf: float, classes=None):
        """Run detection on one image. Returns (annotated_bgr, counts_dict)."""
        results = self.model.predict(
            frame_bgr, conf=conf, imgsz=self.imgsz, device=self.device,
            classes=classes, verbose=False, **self._precision,
        )
        r = results[0]
        return r.plot(), self._counts(r)

    # -- video / camera frame with tracking -------------------------------
    def track_frame(self, frame_bgr, conf: float, classes=None):
        """Run detection + tracking on one frame.
        Returns (annotated_bgr, frame_counts, per_frame_ids).
        per_frame_ids maps class_name -> set of tracker IDs seen this frame, so
        the caller can accumulate unique totals across the whole video."""
        results = self.model.track(
            frame_bgr, conf=conf, imgsz=self.imgsz, device=self.device,
            classes=classes, persist=True, tracker=TRACKER, verbose=False,
            **self._precision,
        )
        r = results[0]
        counts = self._counts(r)
        ids: dict[str, set] = {}
        if r.boxes is not None and r.boxes.id is not None:
            cls_list = r.boxes.cls.int().tolist()
            id_list = r.boxes.id.int().tolist()
            for c, tid in zip(cls_list, id_list):
                ids.setdefault(self.names[c], set()).add(tid)
        return r.plot(), counts, ids

    def reset_tracker(self):
        """Forget tracker state so unique counting restarts cleanly."""
        try:
            if getattr(self.model, "predictor", None) is not None:
                trackers = getattr(self.model.predictor, "trackers", None)
                if trackers:
                    for t in trackers:
                        if hasattr(t, "reset"):
                            t.reset()
        except Exception:
            pass

    def _counts(self, r) -> dict:
        counts: dict[str, int] = {}
        if r.boxes is not None and len(r.boxes) > 0:
            for c in r.boxes.cls.int().tolist():
                name = self.names[c]
                counts[name] = counts.get(name, 0) + 1
        return counts
