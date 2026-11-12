"""
train_window.py - in-app custom object training for Vision Studio.

A guided workflow that lets a non-expert teach the app a brand-new object
(keys, banknotes, a specific product...) without using the command line:

    1. Setup   - name the project, define your custom label(s)
    2. Collect - snap photos from the webcam and/or import existing images
    3. Label   - draw a box around the object in each photo
    4. Train   - one click; YOLO fine-tunes in the background with progress
    5. Use     - the trained model loads straight into Vision Studio

Datasets live in ./datasets/<project>, training output in ./runs.
"""

from __future__ import annotations

import glob
import json
import os
import random
import shutil
import time

import cv2
from PySide6.QtCore import Qt, QThread, Signal, QTimer, QRect
from PySide6.QtGui import QImage, QPainter, QPen, QColor
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLineEdit, QListWidget, QComboBox, QSpinBox, QProgressBar,
    QPlainTextEdit, QFileDialog, QMessageBox, QSizePolicy,
)

HERE = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(HERE, "datasets")
RUNS_DIR = os.path.join(HERE, "runs")
MODELS_DIR = os.path.join(HERE, "models")
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


# =====================================================================
#  Annotation canvas - draw bounding boxes over an image
# =====================================================================
class AnnotationCanvas(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(640, 480)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._qimg: QImage | None = None
        self._iw = self._ih = 0
        self.boxes = []            # [cls, x1, y1, x2, y2] in image pixels
        self.class_names = []
        self.current_class = 0
        self._drawing = False
        self._start = None
        self._cur = None

    # -- image / boxes ------------------------------------------------
    def set_frame(self, bgr):
        """Show a raw frame (used for the live capture preview)."""
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        self._qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        self._iw, self._ih = w, h
        self.boxes = []
        self.update()

    def load(self, path, norm_boxes):
        """Load an image plus its (normalized) YOLO boxes for editing."""
        bgr = cv2.imread(path)
        if bgr is None:
            return
        self.set_frame(bgr)
        boxes = []
        for cls, xc, yc, w, h in norm_boxes:
            x1 = (xc - w / 2) * self._iw
            y1 = (yc - h / 2) * self._ih
            x2 = (xc + w / 2) * self._iw
            y2 = (yc + h / 2) * self._ih
            boxes.append([int(cls), int(x1), int(y1), int(x2), int(y2)])
        self.boxes = boxes
        self.update()

    def yolo_boxes(self):
        """Return current boxes as normalized YOLO tuples."""
        out = []
        if self._iw and self._ih:
            for cls, x1, y1, x2, y2 in self.boxes:
                xc = ((x1 + x2) / 2) / self._iw
                yc = ((y1 + y2) / 2) / self._ih
                w = abs(x2 - x1) / self._iw
                h = abs(y2 - y1) / self._ih
                out.append((cls, xc, yc, w, h))
        return out

    def undo(self):
        if self.boxes:
            self.boxes.pop()
            self.update()

    def clear(self):
        self.boxes = []
        self.update()

    # -- coordinate mapping (image is letterboxed into the widget) ----
    def _geom(self):
        W, H = max(1, self.width()), max(1, self.height())
        if not self._iw:
            return 1.0, 0, 0
        s = min(W / self._iw, H / self._ih)
        ox = (W - self._iw * s) / 2
        oy = (H - self._ih * s) / 2
        return s, ox, oy

    def _to_img(self, pt):
        s, ox, oy = self._geom()
        x = max(0, min(self._iw, (pt.x() - ox) / s))
        y = max(0, min(self._ih, (pt.y() - oy) / s))
        return int(x), int(y)

    # -- mouse --------------------------------------------------------
    def mousePressEvent(self, e):
        if self._qimg and e.button() == Qt.LeftButton:
            self._drawing = True
            self._start = self._to_img(e.position().toPoint())
            self._cur = self._start
            self.update()

    def mouseMoveEvent(self, e):
        if self._drawing:
            self._cur = self._to_img(e.position().toPoint())
            self.update()

    def mouseReleaseEvent(self, e):
        if self._drawing and e.button() == Qt.LeftButton:
            self._drawing = False
            (x1, y1), (x2, y2) = self._start, self._cur
            x1, x2 = sorted((x1, x2))
            y1, y2 = sorted((y1, y2))
            if x2 - x1 > 5 and y2 - y1 > 5:
                self.boxes.append([self.current_class, x1, y1, x2, y2])
            self._start = self._cur = None
            self.update()

    # -- painting -----------------------------------------------------
    @staticmethod
    def _color(cls):
        # private RNG: reseeding the global one here would make the train/val
        # shuffle below predictable
        rng = random.Random(cls * 9973 + 7)
        return QColor(rng.randint(70, 255), rng.randint(70, 255),
                      rng.randint(70, 255))

    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#111"))
        if not self._qimg:
            p.setPen(QColor("#888"))
            p.drawText(self.rect(), Qt.AlignCenter,
                       "Collect photos, then draw a box around your object.")
            return
        s, ox, oy = self._geom()
        p.drawImage(QRect(int(ox), int(oy), int(self._iw * s), int(self._ih * s)),
                    self._qimg)
        for cls, x1, y1, x2, y2 in self.boxes:
            col = self._color(cls)
            p.setPen(QPen(col, 2))
            r = QRect(int(x1 * s + ox), int(y1 * s + oy),
                      int((x2 - x1) * s), int((y2 - y1) * s))
            p.drawRect(r)
            name = self.class_names[cls] if cls < len(self.class_names) else str(cls)
            p.fillRect(QRect(r.x(), r.y() - 16, max(40, len(name) * 8 + 6), 16), col)
            p.setPen(QColor("black"))
            p.drawText(r.x() + 3, r.y() - 3, name)
        if self._drawing and self._start and self._cur:
            col = self._color(self.current_class)
            p.setPen(QPen(col, 2, Qt.DashLine))
            (x1, y1), (x2, y2) = self._start, self._cur
            r = QRect(int(min(x1, x2) * s + ox), int(min(y1, y2) * s + oy),
                      int(abs(x2 - x1) * s), int(abs(y2 - y1) * s))
            p.drawRect(r)


# =====================================================================
#  Training worker (background thread)
# =====================================================================
class TrainWorker(QThread):
    progress = Signal(int, int)   # epoch, total
    log = Signal(str)
    done = Signal(str)            # best.pt path
    failed = Signal(str)

    def __init__(self, data_yaml, epochs, imgsz, device, name):
        super().__init__()
        self.data_yaml = data_yaml
        self.epochs = epochs
        self.imgsz = imgsz
        self.device = device
        self.name = name

    def run(self):
        try:
            from ultralytics import YOLO
            from detector import MODEL_FILES, resolve_weights

            # fine-tune from the nano model of the current YOLO family
            model = YOLO(resolve_weights(MODEL_FILES["nano"]))
            total = self.epochs

            last = {"epoch": 0}

            def on_epoch_end(trainer):
                # Ultralytics fires this once more after the final validation,
                # so report each epoch only once and never past the total.
                try:
                    epoch = min(int(trainer.epoch) + 1, total)
                    if epoch <= last["epoch"]:
                        return
                    last["epoch"] = epoch
                    self.progress.emit(epoch, total)
                    self.log.emit(f"  finished epoch {epoch}/{total}")
                except Exception:
                    pass

            model.add_callback("on_fit_epoch_end", on_epoch_end)
            self.log.emit("Training started. CPU training is slow - "
                          "expect several minutes per epoch on a small dataset.")
            model.train(
                data=self.data_yaml, epochs=self.epochs, imgsz=self.imgsz,
                device=self.device, project=RUNS_DIR, name=self.name,
                exist_ok=True, workers=0, batch=8, plots=False, verbose=False,
            )
            best = os.path.join(RUNS_DIR, self.name, "weights", "best.pt")
            if os.path.isfile(best):
                self.done.emit(best)
            else:
                self.failed.emit("Training finished but best.pt was not found.")
        except Exception as exc:
            self.failed.emit(str(exc))


# =====================================================================
#  Main training window
# =====================================================================
class TrainWindow(QMainWindow):
    def __init__(self, profile, on_model_ready):
        super().__init__()
        self.setWindowTitle("Train Custom Object")
        self.resize(1100, 720)
        self.profile = profile
        self.on_model_ready = on_model_ready

        self.project_dir = None
        self.images_dir = None
        self.labels_dir = None
        self.class_names = []
        self.image_files = []
        self.idx = 0
        self.cap = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._grab)
        self._live = None
        self.worker = None

        os.makedirs(DATASETS_DIR, exist_ok=True)
        self._build_ui()

    # -----------------------------------------------------------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        self.canvas = AnnotationCanvas()
        root.addWidget(self.canvas, stretch=1)

        side = QWidget()
        side.setFixedWidth(340)
        col = QVBoxLayout(side)
        col.setSpacing(10)
        root.addWidget(side)

        col.addWidget(self._setup_group())
        col.addWidget(self._collect_group())
        col.addWidget(self._label_group())
        col.addWidget(self._train_group(), stretch=1)

        self.statusBar().showMessage("Step 1: create a project and add your label(s).")

    def _setup_group(self):
        box = QGroupBox("1. Project & labels")
        lay = QVBoxLayout(box)
        row = QHBoxLayout()
        self.project_name = QLineEdit()
        self.project_name.setPlaceholderText("project name, e.g. keys")
        b = QPushButton("Create / Open")
        b.clicked.connect(self.create_project)
        row.addWidget(self.project_name)
        row.addWidget(b)
        lay.addLayout(row)

        row2 = QHBoxLayout()
        self.new_label = QLineEdit()
        self.new_label.setPlaceholderText("add a label, e.g. key")
        ab = QPushButton("Add")
        ab.clicked.connect(self.add_label)
        row2.addWidget(self.new_label)
        row2.addWidget(ab)
        lay.addLayout(row2)

        self.label_list = QListWidget()
        self.label_list.setMaximumHeight(70)
        lay.addWidget(self.label_list)
        return box

    def _collect_group(self):
        box = QGroupBox("2. Collect photos")
        lay = QVBoxLayout(box)
        row = QHBoxLayout()
        self.b_cam = QPushButton("Start Camera")
        self.b_cam.clicked.connect(self.toggle_camera)
        self.b_snap = QPushButton("Snap")
        self.b_snap.clicked.connect(self.snap)
        self.b_snap.setEnabled(False)
        row.addWidget(self.b_cam)
        row.addWidget(self.b_snap)
        lay.addLayout(row)
        b_imp = QPushButton("Import Images...")
        b_imp.clicked.connect(self.import_images)
        lay.addWidget(b_imp)
        self.count_label = QLabel("0 photos collected")
        lay.addWidget(self.count_label)
        return box

    def _label_group(self):
        box = QGroupBox("3. Label (drag a box around the object)")
        lay = QVBoxLayout(box)
        self.class_combo = QComboBox()
        self.class_combo.currentIndexChanged.connect(
            lambda i: setattr(self.canvas, "current_class", max(0, i)))
        lay.addWidget(self.class_combo)
        row = QHBoxLayout()
        b_begin = QPushButton("Begin / Reload")
        b_begin.clicked.connect(self.begin_labeling)
        b_undo = QPushButton("Undo Box")
        b_undo.clicked.connect(self.canvas.undo)
        b_clear = QPushButton("Clear")
        b_clear.clicked.connect(self.canvas.clear)
        row.addWidget(b_begin)
        row.addWidget(b_undo)
        row.addWidget(b_clear)
        lay.addLayout(row)
        row2 = QHBoxLayout()
        b_prev = QPushButton("< Prev")
        b_prev.clicked.connect(self.prev_image)
        self.idx_label = QLabel("0 / 0")
        self.idx_label.setAlignment(Qt.AlignCenter)
        b_next = QPushButton("Next >")
        b_next.clicked.connect(self.next_image)
        row2.addWidget(b_prev)
        row2.addWidget(self.idx_label)
        row2.addWidget(b_next)
        lay.addLayout(row2)
        return box

    def _train_group(self):
        box = QGroupBox("4. Train & use")
        lay = QVBoxLayout(box)
        row = QHBoxLayout()
        row.addWidget(QLabel("Epochs:"))
        self.epochs = QSpinBox()
        self.epochs.setRange(10, 300)
        self.epochs.setValue(60)
        row.addWidget(self.epochs)
        row.addWidget(QLabel("Image size:"))
        self.imgsz = QComboBox()
        self.imgsz.addItems(["416", "512", "640"])
        row.addWidget(self.imgsz)
        lay.addLayout(row)

        self.b_train = QPushButton("Start Training")
        self.b_train.clicked.connect(self.start_training)
        lay.addWidget(self.b_train)
        self.progress = QProgressBar()
        lay.addWidget(self.progress)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        lay.addWidget(self.log, stretch=1)
        self.b_use = QPushButton("Use this model in Vision Studio")
        self.b_use.setEnabled(False)
        self.b_use.clicked.connect(self.use_model)
        lay.addWidget(self.b_use)
        self._best = None
        return box

    # === Step 1: project & labels ====================================
    def create_project(self):
        name = self.project_name.text().strip().replace(" ", "_")
        if not name:
            QMessageBox.warning(self, "Name needed", "Enter a project name.")
            return
        self.project_dir = os.path.join(DATASETS_DIR, name)
        self.images_dir = os.path.join(self.project_dir, "images")
        self.labels_dir = os.path.join(self.project_dir, "labels")
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.labels_dir, exist_ok=True)
        # load any classes saved previously
        cfile = os.path.join(self.project_dir, "classes.json")
        if os.path.isfile(cfile):
            with open(cfile, encoding="utf-8") as fh:
                self.class_names = json.load(fh)
            self._refresh_classes()
        self._refresh_count()
        self.statusBar().showMessage(f"Project ready: {self.project_dir}")

    def add_label(self):
        if not self.project_dir:
            QMessageBox.warning(self, "Create project first",
                                "Create or open a project before adding labels.")
            return
        name = self.new_label.text().strip()
        if name and name not in self.class_names:
            self.class_names.append(name)
            cfile = os.path.join(self.project_dir, "classes.json")
            with open(cfile, "w", encoding="utf-8") as fh:
                json.dump(self.class_names, fh)
            self._refresh_classes()
        self.new_label.clear()

    def _refresh_classes(self):
        self.label_list.clear()
        self.label_list.addItems(self.class_names)
        self.class_combo.clear()
        self.class_combo.addItems(self.class_names)
        self.canvas.class_names = self.class_names

    # === Step 2: collect =============================================
    def toggle_camera(self):
        if self.cap is None:
            import sys
            backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
            self.cap = cv2.VideoCapture(0, backend)
            if not self.cap.isOpened():
                self.cap = None
                QMessageBox.warning(self, "Camera", "Could not open the camera.")
                return
            self.timer.start(60)
            self.b_cam.setText("Stop Camera")
            self.b_snap.setEnabled(True)
        else:
            self._stop_camera()

    def _stop_camera(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
        self.b_cam.setText("Start Camera")
        self.b_snap.setEnabled(False)

    def _grab(self):
        if self.cap:
            ok, frame = self.cap.read()
            if ok:
                self._live = frame
                self.canvas.set_frame(frame)

    def snap(self):
        if self.images_dir and self._live is not None:
            path = os.path.join(self.images_dir, f"img_{int(time.time()*1000)}.jpg")
            cv2.imwrite(path, self._live)
            self._refresh_count()

    def import_images(self):
        if not self.images_dir:
            QMessageBox.warning(self, "Create project first",
                                "Create a project before importing images.")
            return
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Import images", "", "Images (*.jpg *.jpeg *.png *.bmp *.webp)")
        for p in paths:
            try:
                shutil.copy(p, os.path.join(self.images_dir, os.path.basename(p)))
            except Exception:
                pass
        self._refresh_count()

    def _refresh_count(self):
        n = len(self._list_images()) if self.images_dir else 0
        self.count_label.setText(f"{n} photos collected")

    def _list_images(self):
        files = []
        for ext in IMG_EXTS:
            files += glob.glob(os.path.join(self.images_dir, f"*{ext}"))
        return sorted(files)

    # === Step 3: label ===============================================
    def begin_labeling(self):
        if not self.project_dir:
            QMessageBox.warning(self, "Create project first", "Create a project first.")
            return
        if not self.class_names:
            QMessageBox.warning(self, "Add a label", "Add at least one label first.")
            return
        self._stop_camera()
        self.image_files = self._list_images()
        if not self.image_files:
            QMessageBox.information(self, "No photos", "Collect some photos first.")
            return
        self.canvas.class_names = self.class_names
        self.canvas.current_class = max(0, self.class_combo.currentIndex())
        self.idx = 0
        self._load_current()

    def _label_path(self, img_path):
        base = os.path.splitext(os.path.basename(img_path))[0]
        return os.path.join(self.labels_dir, base + ".txt")

    def _read_label(self, img_path):
        boxes = []
        lp = self._label_path(img_path)
        if os.path.isfile(lp):
            with open(lp) as fh:
                for line in fh:
                    parts = line.split()
                    if len(parts) == 5:
                        c, xc, yc, w, h = parts
                        boxes.append([int(float(c)), float(xc), float(yc),
                                      float(w), float(h)])
        return boxes

    def _save_current(self):
        if not self.image_files:
            return
        img = self.image_files[self.idx]
        with open(self._label_path(img), "w") as f:
            for cls, xc, yc, w, h in self.canvas.yolo_boxes():
                f.write(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

    def _load_current(self):
        img = self.image_files[self.idx]
        self.canvas.current_class = max(0, self.class_combo.currentIndex())
        self.canvas.load(img, self._read_label(img))
        self.idx_label.setText(f"{self.idx + 1} / {len(self.image_files)}")
        self.statusBar().showMessage(
            f"Labeling {os.path.basename(img)} - drag a box, then Next.")

    def next_image(self):
        if not self.image_files:
            return
        self._save_current()
        self.idx = min(self.idx + 1, len(self.image_files) - 1)
        self._load_current()

    def prev_image(self):
        if not self.image_files:
            return
        self._save_current()
        self.idx = max(self.idx - 1, 0)
        self._load_current()

    # === Step 4: train ===============================================
    def _build_yaml(self):
        imgs = self._list_images()
        labeled = [f for f in imgs if os.path.isfile(self._label_path(f))]
        if len(labeled) < 2:
            raise ValueError(
                "Label at least 2 photos before training (more is much better).")
        random.shuffle(labeled)
        n_val = max(1, int(len(labeled) * 0.2))
        val, train = labeled[:n_val], labeled[n_val:]
        if not train:
            train = val
        with open(os.path.join(self.project_dir, "train.txt"), "w") as f:
            f.write("\n".join(os.path.abspath(p) for p in train))
        with open(os.path.join(self.project_dir, "val.txt"), "w") as f:
            f.write("\n".join(os.path.abspath(p) for p in val))
        yaml_path = os.path.join(self.project_dir, "data.yaml")
        with open(yaml_path, "w", encoding="utf-8") as f:
            # JSON strings are valid YAML: keeps backslashes, spaces and '#'
            # in Windows paths from being misread.
            f.write(f"path: {json.dumps(os.path.abspath(self.project_dir))}\n")
            f.write("train: train.txt\n")
            f.write("val: val.txt\n")
            f.write(f"names: {json.dumps(self.class_names)}\n")
        return yaml_path, len(train), len(val)

    def start_training(self):
        if not self.project_dir or not self.class_names:
            QMessageBox.warning(self, "Not ready", "Create a project and labels first.")
            return
        self._save_current()
        try:
            yaml_path, n_train, n_val = self._build_yaml()
        except Exception as exc:
            QMessageBox.warning(self, "Cannot train", str(exc))
            return
        self.log.appendPlainText(
            f"Dataset: {n_train} train / {n_val} val images, "
            f"{len(self.class_names)} class(es).")
        self.b_train.setEnabled(False)
        self.progress.setRange(0, self.epochs.value())
        self.progress.setValue(0)
        self.worker = TrainWorker(
            yaml_path, self.epochs.value(), int(self.imgsz.currentText()),
            self.profile.device, os.path.basename(self.project_dir))
        self.worker.progress.connect(self._on_progress)
        self.worker.log.connect(self.log.appendPlainText)
        self.worker.done.connect(self._on_done)
        self.worker.failed.connect(self._on_failed)
        self.worker.start()
        self.statusBar().showMessage("Training... you can watch progress below.")

    def _on_progress(self, epoch, total):
        self.progress.setValue(epoch)

    def _on_done(self, best):
        # keep a copy in models/ so it is easy to reuse later
        os.makedirs(MODELS_DIR, exist_ok=True)
        dest = os.path.join(MODELS_DIR,
                            f"custom_{os.path.basename(self.project_dir)}.pt")
        try:
            shutil.copy(best, dest)
        except Exception:
            dest = best
        self._best = dest
        self.b_train.setEnabled(True)
        self.b_use.setEnabled(True)
        self.log.appendPlainText(f"DONE. Model saved:\n{dest}")
        self.statusBar().showMessage("Training complete!")
        QMessageBox.information(self, "Training complete",
                                "Your custom model is ready. Click "
                                "'Use this model in Vision Studio'.")

    def _on_failed(self, msg):
        self.b_train.setEnabled(True)
        self.log.appendPlainText("ERROR: " + msg)
        QMessageBox.critical(self, "Training failed", msg)

    def use_model(self):
        if self._best and self.on_model_ready:
            self.on_model_ready(self._best)
            self.statusBar().showMessage("Loaded into Vision Studio.")

    def closeEvent(self, event):
        self._stop_camera()
        if self.worker and self.worker.isRunning():
            QMessageBox.information(self, "Training running",
                                    "Training continues in the background; "
                                    "this window will stay open.")
            event.ignore()
            return
        super().closeEvent(event)
