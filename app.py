"""
Vision Studio - offline computer-vision object detection & counting.

A Windows desktop app (PySide6 + Ultralytics YOLO) that detects and counts
objects in images, video files, and the live camera. It auto-detects your
hardware and picks the best model + resolution so it runs well on any machine.

Run with:  run.bat   (or  python app.py  inside the venv)
"""

from __future__ import annotations

import os
import sys
import time
from collections import defaultdict

import cv2
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QImage, QPixmap, QFont
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QGroupBox, QSlider, QComboBox, QListWidget, QListWidgetItem,
    QLineEdit, QTableWidget, QTableWidgetItem, QFileDialog, QMessageBox,
    QHeaderView, QSizePolicy,
)

from detector import Detector, HardwareProfile, MODEL_FAMILY

# Quick class presets (COCO names).
PRESETS = {
    "People": ["person"],
    "Vehicles": ["car", "motorcycle", "bus", "truck", "bicycle"],
    "Phones": ["cell phone"],
}

# Model dropdown entries: (label, detector key). None = resolved at runtime.
MODEL_CHOICES = [
    ("Auto (recommended)", None),
    ("Nano - fastest", "nano"),
    ("Small - balanced", "small"),
    ("Medium - accurate", "medium"),
    ("Large - most accurate", "large"),
]
CUSTOM_MODEL_INDEX = len(MODEL_CHOICES)      # "Custom model (.pt)..." entry


def bgr_to_qimage(frame) -> QImage:
    """Convert an OpenCV BGR numpy frame to a detached QImage."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    return QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()


class InferenceWorker(QThread):
    """Runs capture + detection/tracking on a background thread so the UI
    stays responsive. Works for both video files and the live camera."""

    frame_ready = Signal(object, object, object, float)  # qimage, counts, totals, fps
    ended = Signal()
    failed = Signal(str)

    def __init__(self, detector: Detector, source, is_camera: bool,
                 conf: float, classes):
        super().__init__()
        self.detector = detector
        self.source = source
        self.is_camera = is_camera
        self.conf = conf
        self.classes = classes
        self._stop = False
        self._unique = defaultdict(set)

    def set_conf(self, conf: float):
        self.conf = conf

    def set_classes(self, classes):
        self.classes = classes

    def stop(self):
        self._stop = True

    def run(self):
        cap = None
        try:
            if self.is_camera:
                # DirectShow opens webcams much faster on Windows.
                backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
                cap = cv2.VideoCapture(self.source, backend)
            else:
                cap = cv2.VideoCapture(self.source)
            if not cap.isOpened():
                self.failed.emit("Could not open the video source.")
                return

            self.detector.reset_tracker()
            last = time.time()
            fps = 0.0
            while not self._stop:
                ok, frame = cap.read()
                if not ok:
                    break  # end of video (or camera dropped)

                annotated, counts, ids = self.detector.track_frame(
                    frame, self.conf, self.classes)

                for name, id_set in ids.items():
                    self._unique[name].update(id_set)
                totals = {n: len(s) for n, s in self._unique.items()}

                now = time.time()
                dt = now - last
                last = now
                if dt > 0:
                    fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else (1.0 / dt)

                self.frame_ready.emit(
                    bgr_to_qimage(annotated), counts, totals, fps)

            self.ended.emit()
        except Exception as exc:  # surface errors to the UI instead of crashing
            self.failed.emit(str(exc))
        finally:
            if cap is not None:
                cap.release()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(
            f"Vision Studio - Object Detection & Counting "
            f"({MODEL_FAMILY.upper()})")
        self.resize(1280, 880)

        # --- detection engine --------------------------------------------
        self.profile = HardwareProfile.detect()
        self.detector = Detector(self.profile.recommended_model, self.profile)
        self.name_to_id = {v: k for k, v in self.detector.names.items()}

        self.worker: InferenceWorker | None = None
        self.current_source = None       # ("image"|"video"|"camera", value)
        self._last_qimage: QImage | None = None

        self._build_ui()
        self._populate_classes()
        self.statusBar().showMessage(self._model_status())

    # ===================================================================
    #  UI construction
    # ===================================================================
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        # ---- left: image/video display ----
        self.view = QLabel("Open an image or video, or start the camera.")
        self.view.setAlignment(Qt.AlignCenter)
        self.view.setMinimumSize(640, 480)
        self.view.setStyleSheet(
            "background:#111; color:#888; border:1px solid #333;")
        self.view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        root.addWidget(self.view, stretch=1)

        # ---- right: control panel ----
        panel_w = QWidget()
        panel_w.setFixedWidth(330)
        panel = QVBoxLayout(panel_w)
        panel.setSpacing(10)
        root.addWidget(panel_w)

        panel.addWidget(self._source_group())
        panel.addWidget(self._hardware_group())
        panel.addWidget(self._detection_group())
        panel.addWidget(self._classes_group())
        panel.addWidget(self._counts_group(), stretch=1)
        panel.addWidget(self._custom_group())

    def _source_group(self) -> QGroupBox:
        box = QGroupBox("Source")
        lay = QVBoxLayout(box)
        row1 = QHBoxLayout()
        b_img = QPushButton("Open Image")
        b_vid = QPushButton("Open Video")
        b_img.clicked.connect(self.open_image)
        b_vid.clicked.connect(self.open_video)
        row1.addWidget(b_img)
        row1.addWidget(b_vid)
        lay.addLayout(row1)

        row2 = QHBoxLayout()
        self.cam_combo = QComboBox()
        self.cam_combo.addItems(["Camera 0", "Camera 1", "Camera 2"])
        b_cam = QPushButton("Start Camera")
        b_cam.clicked.connect(self.start_camera)
        row2.addWidget(self.cam_combo)
        row2.addWidget(b_cam)
        lay.addLayout(row2)

        row3 = QHBoxLayout()
        self.b_stop = QPushButton("Stop")
        self.b_stop.clicked.connect(self.stop)
        self.b_stop.setEnabled(False)
        b_save = QPushButton("Save Result")
        b_save.clicked.connect(self.save_result)
        row3.addWidget(self.b_stop)
        row3.addWidget(b_save)
        lay.addLayout(row3)
        return box

    def _hardware_group(self) -> QGroupBox:
        box = QGroupBox("Hardware (auto-optimized)")
        lay = QVBoxLayout(box)
        self.hw_label = QLabel(self.profile.summary)
        self.hw_label.setWordWrap(True)
        self.hw_label.setStyleSheet("color:#2e7d32;")
        lay.addWidget(self.hw_label)
        self.model_label = QLabel()
        self.model_label.setWordWrap(True)
        self.model_label.setStyleSheet("color:#888;")
        lay.addWidget(self.model_label)
        self._refresh_model_label()
        return box

    def _model_status(self) -> str:
        """One-line 'which model is loaded' text for the status bar."""
        text = (f"Model: {os.path.basename(self.detector.weights)} | "
                f"{self.profile.summary}")
        if self.detector.note:
            text += f" | NOTE: {self.detector.note}"
        return text

    def _refresh_model_label(self):
        text = f"Model in use: {os.path.basename(self.detector.weights)}"
        if self.detector.note:
            text += "\n" + self.detector.note
        self.model_label.setText(text)

    def _detection_group(self) -> QGroupBox:
        box = QGroupBox("Detection")
        lay = QVBoxLayout(box)

        self.conf_label = QLabel("Confidence: 35%")
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(10, 95)
        self.conf_slider.setValue(35)
        self.conf_slider.valueChanged.connect(self.on_conf_change)
        lay.addWidget(self.conf_label)
        lay.addWidget(self.conf_slider)

        lay.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(
            [label for label, _ in MODEL_CHOICES] + ["Custom model (.pt)..."])
        self.model_combo.currentIndexChanged.connect(self.on_model_change)
        lay.addWidget(self.model_combo)
        return box

    def _classes_group(self) -> QGroupBox:
        box = QGroupBox("Classes / Labels")
        lay = QVBoxLayout(box)

        presets = QHBoxLayout()
        for name in ["People", "Vehicles", "Phones"]:
            b = QPushButton(name)
            b.clicked.connect(lambda _=False, n=name: self.apply_preset(n))
            presets.addWidget(b)
        lay.addLayout(presets)

        presets2 = QHBoxLayout()
        b_all = QPushButton("All")
        b_none = QPushButton("None")
        b_all.clicked.connect(lambda: self.set_all_classes(True))
        b_none.clicked.connect(lambda: self.set_all_classes(False))
        presets2.addWidget(b_all)
        presets2.addWidget(b_none)
        lay.addLayout(presets2)

        self.class_search = QLineEdit()
        self.class_search.setPlaceholderText("Search labels...")
        self.class_search.textChanged.connect(self.filter_classes)
        lay.addWidget(self.class_search)

        self.class_list = QListWidget()
        self.class_list.setMinimumHeight(84)
        self.class_list.setMaximumHeight(150)
        self.class_list.itemChanged.connect(self.on_classes_changed)
        lay.addWidget(self.class_list)
        return box

    def _counts_group(self) -> QGroupBox:
        box = QGroupBox("Counts")
        lay = QVBoxLayout(box)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Label", "In view", "Total seen"])
        self.table.setToolTip(
            "In view = objects on screen now.\n"
            "Total seen = unique objects tracked over the whole video/camera "
            "session (e.g. how many distinct cars passed).")
        self.table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        lay.addWidget(self.table)

        self.total_label = QLabel("Total objects in view: 0")
        f = QFont()
        f.setBold(True)
        self.total_label.setFont(f)
        lay.addWidget(self.total_label)

        b_reset = QPushButton("Reset Counts")
        b_reset.clicked.connect(self.reset_counts)
        lay.addWidget(b_reset)
        return box

    def _custom_group(self) -> QGroupBox:
        box = QGroupBox("Custom training")
        lay = QVBoxLayout(box)
        b = QPushButton("Train Custom Object...")
        b.clicked.connect(self.open_trainer)
        lay.addWidget(b)
        return box

    def open_trainer(self):
        from train_window import TrainWindow
        self._trainer = TrainWindow(self.profile, self.use_custom_model)
        self._trainer.show()

    def use_custom_model(self, path: str):
        self._load_model(path)   # also rebuilds the label list if it changed
        self.model_combo.blockSignals(True)
        self.model_combo.setCurrentIndex(CUSTOM_MODEL_INDEX)
        self.model_combo.blockSignals(False)
        QMessageBox.information(
            self, "Custom model loaded",
            f"Now detecting with: {os.path.basename(path)}\n"
            f"Labels: {', '.join(self.detector.names.values())}")

    def _populate_classes(self):
        self.class_list.blockSignals(True)
        for cid in sorted(self.detector.names):
            name = self.detector.names[cid]
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            item.setData(Qt.UserRole, cid)
            self.class_list.addItem(item)
        self.class_list.blockSignals(False)

    # ===================================================================
    #  Class selection helpers
    # ===================================================================
    def selected_class_ids(self):
        """Return list of checked class IDs, or None when all/none are checked
        (None = detect everything)."""
        checked = []
        total = self.class_list.count()
        for i in range(total):
            it = self.class_list.item(i)
            if it.checkState() == Qt.Checked:
                checked.append(it.data(Qt.UserRole))
        if not checked or len(checked) == total:
            return None
        return checked

    def apply_preset(self, name: str):
        wanted = set(PRESETS.get(name, []))
        self.class_list.blockSignals(True)
        for i in range(self.class_list.count()):
            it = self.class_list.item(i)
            it.setCheckState(Qt.Checked if it.text() in wanted else Qt.Unchecked)
        self.class_list.blockSignals(False)
        self.on_classes_changed()

    def set_all_classes(self, checked: bool):
        state = Qt.Checked if checked else Qt.Unchecked
        self.class_list.blockSignals(True)
        for i in range(self.class_list.count()):
            self.class_list.item(i).setCheckState(state)
        self.class_list.blockSignals(False)
        self.on_classes_changed()

    def filter_classes(self, text: str):
        text = text.lower()
        for i in range(self.class_list.count()):
            it = self.class_list.item(i)
            it.setHidden(text not in it.text().lower())

    def on_classes_changed(self, *_):
        classes = self.selected_class_ids()
        if self.worker:
            self.worker.set_classes(classes)
        elif self.current_source and self.current_source[0] == "image":
            self.rerun_image()

    # ===================================================================
    #  Detection settings
    # ===================================================================
    def on_conf_change(self, value: int):
        self.conf_label.setText(f"Confidence: {value}%")
        if self.worker:
            self.worker.set_conf(value / 100.0)
        elif self.current_source and self.current_source[0] == "image":
            self.rerun_image()

    def on_model_change(self, index: int):
        if 0 <= index < len(MODEL_CHOICES):
            key = MODEL_CHOICES[index][1] or self.profile.recommended_model
            self._load_model(key)
        elif index == CUSTOM_MODEL_INDEX:
            path, _ = QFileDialog.getOpenFileName(
                self, "Select a trained model", "", "YOLO model (*.pt)")
            if path:
                self._load_model(path)
            else:
                self.model_combo.setCurrentIndex(0)

    def _load_model(self, key_or_path: str):
        was = self.current_source
        self.stop()
        self.statusBar().showMessage(
            "Loading model... (the first use of a size downloads it once)")
        QApplication.processEvents()
        old_names = dict(self.detector.names)
        try:
            self.detector = Detector(key_or_path, self.profile)
            self.name_to_id = {v: k for k, v in self.detector.names.items()}
        except Exception as exc:
            QMessageBox.critical(self, "Model error",
                                 f"Could not load model:\n{exc}")
            self.statusBar().showMessage(self._model_status())
            return
        if dict(self.detector.names) != old_names:
            # different label set (e.g. custom model <-> built-in): rebuild list
            self.class_list.clear()
            self._populate_classes()
        self._refresh_model_label()
        self.statusBar().showMessage(self._model_status())
        # resume the previous source with the new model
        if was:
            kind, val = was
            if kind == "image":
                self.rerun_image()
            elif kind == "video":
                self._start_worker(val, is_camera=False)
            elif kind == "camera":
                self._start_worker(val, is_camera=True)

    # ===================================================================
    #  Sources
    # ===================================================================
    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open image", "",
            "Images (*.jpg *.jpeg *.png *.bmp *.webp)")
        if not path:
            return
        self.stop()
        self.current_source = ("image", path)
        self.rerun_image()

    def rerun_image(self):
        if not self.current_source or self.current_source[0] != "image":
            return
        path = self.current_source[1]
        frame = cv2.imread(path)
        if frame is None:
            QMessageBox.warning(self, "Error", "Could not read that image.")
            return
        conf = self.conf_slider.value() / 100.0
        annotated, counts = self.detector.predict_image(
            frame, conf, self.selected_class_ids())
        self._last_qimage = bgr_to_qimage(annotated)
        self._show_qimage(self._last_qimage)
        self._update_counts(counts, {})

    def open_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open video", "",
            "Videos (*.mp4 *.avi *.mov *.mkv *.wmv)")
        if not path:
            return
        self._start_worker(path, is_camera=False)

    def start_camera(self):
        index = self.cam_combo.currentIndex()
        self._start_worker(index, is_camera=True)

    def _start_worker(self, source, is_camera: bool):
        self.stop()
        self.current_source = ("camera" if is_camera else "video", source)
        conf = self.conf_slider.value() / 100.0
        self.worker = InferenceWorker(
            self.detector, source, is_camera, conf, self.selected_class_ids())
        self.worker.frame_ready.connect(self.on_frame)
        self.worker.ended.connect(self.on_ended)
        self.worker.failed.connect(self.on_failed)
        self.worker.start()
        self.b_stop.setEnabled(True)
        self.statusBar().showMessage("Running...")

    def stop(self):
        if self.worker:
            self.worker.stop()
            self.worker.wait(3000)
            self.worker = None
        self.b_stop.setEnabled(False)

    # ===================================================================
    #  Frame / counts display
    # ===================================================================
    def on_frame(self, qimage, counts, totals, fps):
        self._last_qimage = qimage
        self._show_qimage(qimage)
        self._update_counts(counts, totals)
        self.statusBar().showMessage(f"Running... {fps:4.1f} FPS")

    def on_ended(self):
        self.b_stop.setEnabled(False)
        self.statusBar().showMessage("Finished.")
        self.worker = None

    def on_failed(self, msg):
        self.b_stop.setEnabled(False)
        self.worker = None
        QMessageBox.warning(self, "Source error", msg)
        self.statusBar().showMessage("Error: " + msg)

    def _show_qimage(self, qimage: QImage):
        pix = QPixmap.fromImage(qimage).scaled(
            self.view.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.view.setPixmap(pix)

    def _update_counts(self, counts: dict, totals: dict):
        names = sorted(set(counts) | set(totals))
        self.table.setRowCount(len(names))
        for row, name in enumerate(names):
            self.table.setItem(row, 0, QTableWidgetItem(name))
            self.table.setItem(row, 1, QTableWidgetItem(str(counts.get(name, 0))))
            tot = totals.get(name, "")
            self.table.setItem(row, 2, QTableWidgetItem(str(tot) if tot != "" else "-"))
        self.total_label.setText(
            f"Total objects in view: {sum(counts.values())}")

    def reset_counts(self):
        if self.worker:
            self.worker._unique.clear()
            self.detector.reset_tracker()
        self.table.setRowCount(0)
        self.total_label.setText("Total objects in view: 0")

    def save_result(self):
        if self._last_qimage is None:
            QMessageBox.information(self, "Nothing to save",
                                    "Run a detection first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save annotated image", "result.png", "PNG image (*.png)")
        if path:
            self._last_qimage.save(path)
            self.statusBar().showMessage("Saved: " + path)

    # keep the displayed frame scaled to the window
    def resizeEvent(self, event):
        if self._last_qimage is not None:
            self._show_qimage(self._last_qimage)
        super().resizeEvent(event)

    def closeEvent(self, event):
        self.stop()
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
