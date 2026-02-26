"""
Assembly Line Operator GUI — Screw Detection

Workflow:
  Fill form → Count Screws (confirms 28 on desk) → Install screws → Submit

Run: python run_gui.py

Uses PyQt6 for GUI (PySide6 has platform plugin issues on macOS).
"""

import sys
import os
import cv2
import time
from collections import deque
from datetime import datetime
from PIL import Image

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QLineEdit, QComboBox, QPushButton, QProgressBar,
    QGroupBox, QFormLayout, QSizePolicy,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal as Signal
from PyQt6.QtGui import QImage, QPixmap, QFont

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH      = os.path.join(SCRIPT_DIR, "models", "rfdetr_96_108.pt")
EXPECTED_SCREWS = 28
CONSEC_FRAMES   = 5      # consecutive inference frames required to pass a gate
FRAME_INTERVAL  = 0.3    # seconds between inference calls
CONF_THRESHOLD  = 0.50
CROP_SIZE       = 704

HW_REVISIONS = ["09a", "09b", "09c", "09d", "09e"]
OPERATORS    = ["A", "B", "C", "D"]

# App states
FORM         = "FORM"
COUNT_CHECK  = "COUNT_CHECK"
BUILDING     = "BUILDING"
SUBMIT_READY = "SUBMIT_READY"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def live_crop(frame):
    """Direct center slice at CROP_SIZE × CROP_SIZE — matches live_mode in run_inference.py."""
    h, w = frame.shape[:2]
    cx1 = (w - CROP_SIZE) // 2
    cy1 = (h - CROP_SIZE) // 2
    return frame[cy1:cy1 + CROP_SIZE, cx1:cx1 + CROP_SIZE]


def predict_screw_count(model, frame):
    """Return number of screws detected in the center crop of frame."""
    crop = live_crop(frame)
    pil  = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    dets = model.predict(pil, threshold=CONF_THRESHOLD)
    return len(dets) if dets is not None else 0


def frame_to_qimage(frame):
    """Convert BGR numpy frame to QImage with red crop-region rectangle."""
    display = frame.copy()
    h, w = display.shape[:2]
    cx1 = (w - CROP_SIZE) // 2
    cy1 = (h - CROP_SIZE) // 2
    cv2.rectangle(display, (cx1, cy1), (cx1 + CROP_SIZE, cy1 + CROP_SIZE), (0, 0, 255), 2)
    rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
    rh, rw, ch = rgb.shape
    return QImage(rgb.data, rw, rh, ch * rw, QImage.Format.Format_RGB888).copy()


# ---------------------------------------------------------------------------
# Model Loader Thread
# ---------------------------------------------------------------------------
class ModelLoader(QThread):
    model_ready = Signal(object)
    load_error  = Signal(str)

    def run(self):
        try:
            from rfdetr import RFDETRLarge
            model = RFDETRLarge(pretrain_weights=MODEL_PATH, resolution=CROP_SIZE, num_classes=2)
            model.optimize_for_inference()
            self.model_ready.emit(model)
        except Exception as exc:
            self.load_error.emit(str(exc))


# ---------------------------------------------------------------------------
# Camera + Inference Worker Thread
# ---------------------------------------------------------------------------
class CameraWorker(QThread):
    frame_ready  = Signal(QImage)
    count_update = Signal(int)
    gate_passed  = Signal(str)   # emits "start" or "end"

    def __init__(self, model, gate_mode):
        """
        gate_mode:
            "start" — passes when CONSEC_FRAMES consecutive counts >= EXPECTED_SCREWS
            "end"   — passes when CONSEC_FRAMES consecutive counts == 0
        """
        super().__init__()
        self.model      = model
        self.gate_mode  = gate_mode
        self._running   = True
        self._consec    = deque(maxlen=CONSEC_FRAMES)
        self._gate_done = False

    def stop(self):
        self._running = False

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        last_infer = 0.0

        while self._running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            # Emit every frame for live display (~camera FPS)
            self.frame_ready.emit(frame_to_qimage(frame))

            # Run inference every FRAME_INTERVAL seconds
            now = time.time()
            if now - last_infer >= FRAME_INTERVAL:
                last_infer = now
                count = predict_screw_count(self.model, frame)
                self.count_update.emit(count)

                if not self._gate_done:
                    self._consec.append(count)
                    if len(self._consec) == CONSEC_FRAMES:
                        if self.gate_mode == "start" and all(c >= EXPECTED_SCREWS for c in self._consec):
                            self._gate_done = True
                            self.gate_passed.emit("start")
                        elif self.gate_mode == "end" and all(c == 0 for c in self._consec):
                            self._gate_done = True
                            self.gate_passed.emit("end")

        cap.release()


# ---------------------------------------------------------------------------
# Main Window
# ---------------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Screw Detection — Assembly Line")
        self.resize(1400, 800)

        self.model         = None
        self.state         = FORM
        self.camera_worker = None
        self._model_loader = None

        self._build_ui()
        self._load_model()

    # ── UI Construction ──────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(16)
        root.addWidget(self._build_left_panel(),  stretch=0)
        root.addWidget(self._build_right_panel(), stretch=1)

    def _build_left_panel(self):
        panel = QWidget()
        panel.setFixedWidth(300)
        layout = QVBoxLayout(panel)
        layout.setSpacing(16)
        layout.setContentsMargins(0, 0, 0, 0)

        title = QLabel("Assembly Line\nOperator")
        title.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Build info form
        form_box = QGroupBox("Build Info")
        form = QFormLayout(form_box)
        form.setSpacing(10)

        self.serial_edit    = QLineEdit()
        self.serial_edit.setPlaceholderText("Scan or enter serial #")
        self.rev_combo      = QComboBox()
        self.rev_combo.addItems(HW_REVISIONS)
        self.operator_combo = QComboBox()
        self.operator_combo.addItems(OPERATORS)
        self.checked_combo  = QComboBox()
        self.checked_combo.addItems(OPERATORS)

        form.addRow("Serial #:",    self.serial_edit)
        form.addRow("HW Revision:", self.rev_combo)
        form.addRow("Operator:",    self.operator_combo)
        form.addRow("Checked By:",  self.checked_combo)
        layout.addWidget(form_box)

        # Buttons
        self.btn_count = QPushButton("Count Screws")
        self.btn_count.setMinimumHeight(50)
        self.btn_count.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.btn_count.setEnabled(False)
        self.btn_count.clicked.connect(self._on_count_screws)

        self.btn_bypass = QPushButton("Bypass Camera Check")
        self.btn_bypass.setMinimumHeight(36)
        self.btn_bypass.setEnabled(False)
        self.btn_bypass.clicked.connect(self._on_bypass)

        self.btn_submit = QPushButton("Submit Build")
        self.btn_submit.setMinimumHeight(50)
        self.btn_submit.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.btn_submit.setEnabled(False)
        self.btn_submit.setStyleSheet(
            "QPushButton:enabled { background-color: #2e7d32; color: white; border-radius: 4px; }"
        )
        self.btn_submit.clicked.connect(self._on_submit)

        layout.addWidget(self.btn_count)
        layout.addWidget(self.btn_bypass)
        layout.addSpacing(8)
        layout.addWidget(self.btn_submit)
        layout.addStretch()

        self.model_status = QLabel("Loading model...")
        self.model_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.model_status.setStyleSheet("color: orange; font-size: 11px;")
        layout.addWidget(self.model_status)

        return panel

    def _build_right_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(12)
        layout.setContentsMargins(0, 0, 0, 0)

        self.camera_label = QLabel("Camera inactive")
        self.camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_label.setStyleSheet(
            "background-color: #1a1a1a; color: #666666; border-radius: 6px; font-size: 14px;"
        )
        self.camera_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.camera_label.setMinimumHeight(450)
        layout.addWidget(self.camera_label, stretch=1)

        self.count_label = QLabel("Screws on desk: — / 28")
        self.count_label.setFont(QFont("Arial", 13))
        layout.addWidget(self.count_label)

        # Progress bar value = installed screws (fills up in both phases)
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(EXPECTED_SCREWS)
        self.progress_bar.setValue(0)
        self.progress_bar.setMinimumHeight(26)
        self.progress_bar.setFormat(f"0 / {EXPECTED_SCREWS}")
        layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Fill in build info and click Count Screws.")
        self.status_label.setWordWrap(True)
        self.status_label.setFont(QFont("Arial", 11))
        self.status_label.setStyleSheet("color: #555555;")
        layout.addWidget(self.status_label)

        return panel

    # ── Model Loading ─────────────────────────────────────────────────────────

    def _load_model(self):
        self._model_loader = ModelLoader()
        self._model_loader.model_ready.connect(self._on_model_ready)
        self._model_loader.load_error.connect(self._on_model_error)
        self._model_loader.start()

    def _on_model_ready(self, model):
        self.model = model
        self.model_status.setText("Model ready")
        self.model_status.setStyleSheet("color: green; font-size: 11px;")
        self._apply_state(FORM)

    def _on_model_error(self, msg):
        self.model_status.setText("Model error — check console")
        self.model_status.setStyleSheet("color: red; font-size: 11px;")
        print(f"Model load error: {msg}")

    # ── State Machine ─────────────────────────────────────────────────────────

    def _apply_state(self, new_state):
        self.state = new_state

        # Form fields editable only at FORM
        editable = (new_state == FORM)
        for w in [self.serial_edit, self.rev_combo, self.operator_combo, self.checked_combo]:
            w.setEnabled(editable)

        self.btn_count.setEnabled(new_state == FORM and self.model is not None)
        self.btn_bypass.setEnabled(new_state in (FORM, COUNT_CHECK))
        self.btn_submit.setEnabled(new_state == SUBMIT_READY)

        # Camera management
        if new_state == COUNT_CHECK:
            self._start_camera("start")
        elif new_state == BUILDING:
            self._start_camera("end")
        else:
            self._stop_camera()
            self.camera_label.setPixmap(QPixmap())
            self.camera_label.setText("Camera inactive")

        # Status messages
        messages = {
            FORM:         "Fill in build info and click Count Screws.",
            COUNT_CHECK:  f"Place all {EXPECTED_SCREWS} screws on the desk. Spread them out to avoid overlap.",
            BUILDING:     "Install screws one by one. Submit unlocks when the desk is clear.",
            SUBMIT_READY: f"All {EXPECTED_SCREWS} screws installed. Press Submit to complete the build.",
        }
        self.status_label.setText(messages[new_state])

        # Reset count display per state
        if new_state == FORM:
            self.count_label.setText(f"Screws on desk: — / {EXPECTED_SCREWS}")
            self.progress_bar.setValue(0)
            self.progress_bar.setFormat(f"0 / {EXPECTED_SCREWS}")
        elif new_state == BUILDING:
            self.count_label.setText(f"Installed: 0 / {EXPECTED_SCREWS}  |  {EXPECTED_SCREWS} remaining on desk")
            self.progress_bar.setValue(0)
            self.progress_bar.setFormat(f"0 / {EXPECTED_SCREWS} installed")
        elif new_state == SUBMIT_READY:
            self.count_label.setText(f"Installed: {EXPECTED_SCREWS} / {EXPECTED_SCREWS}  |  0 remaining on desk")
            self.progress_bar.setValue(EXPECTED_SCREWS)
            self.progress_bar.setFormat(f"{EXPECTED_SCREWS} / {EXPECTED_SCREWS} installed")

    # ── Camera ────────────────────────────────────────────────────────────────

    def _start_camera(self, gate_mode):
        self._stop_camera()
        self.camera_worker = CameraWorker(self.model, gate_mode)
        self.camera_worker.frame_ready.connect(self._on_frame)
        self.camera_worker.count_update.connect(self._on_count_update)
        self.camera_worker.gate_passed.connect(self._on_gate_passed)
        self.camera_worker.start()

    def _stop_camera(self):
        if self.camera_worker is not None:
            self.camera_worker.stop()
            self.camera_worker.wait(3000)
            self.camera_worker = None

    # ── Camera Signals ────────────────────────────────────────────────────────

    def _on_frame(self, qimage):
        pixmap = QPixmap.fromImage(qimage)
        self.camera_label.setPixmap(
            pixmap.scaled(
                self.camera_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def _on_count_update(self, count):
        if self.state == COUNT_CHECK:
            val = min(count, EXPECTED_SCREWS)
            self.count_label.setText(f"Screws on desk: {count} / {EXPECTED_SCREWS}")
            self.progress_bar.setValue(val)
            self.progress_bar.setFormat(f"{val} / {EXPECTED_SCREWS}")

        elif self.state == BUILDING:
            remaining = count
            installed = max(0, EXPECTED_SCREWS - remaining)
            self.count_label.setText(
                f"Installed: {installed} / {EXPECTED_SCREWS}  |  {remaining} remaining on desk"
            )
            self.progress_bar.setValue(installed)
            self.progress_bar.setFormat(f"{installed} / {EXPECTED_SCREWS} installed")

    def _on_gate_passed(self, gate):
        if gate == "start" and self.state == COUNT_CHECK:
            self._apply_state(BUILDING)
        elif gate == "end" and self.state == BUILDING:
            self._apply_state(SUBMIT_READY)

    # ── Button Handlers ───────────────────────────────────────────────────────

    def _on_count_screws(self):
        self._apply_state(COUNT_CHECK)

    def _on_bypass(self):
        print("BYPASS: Camera check skipped by operator.")
        self._apply_state(BUILDING)

    def _on_submit(self):
        serial   = self.serial_edit.text().strip() or "(none)"
        rev      = self.rev_combo.currentText()
        operator = self.operator_combo.currentText()
        checked  = self.checked_combo.currentText()
        ts       = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print()
        print("--- BUILD SUBMITTED ---")
        print(f"Serial #:     {serial}")
        print(f"HW Revision:  {rev}")
        print(f"Operator:     {operator}")
        print(f"Checked By:   {checked}")
        print(f"Timestamp:    {ts}")
        print("Status:       uploaded to database")
        print()

        # Reset for next build
        self.serial_edit.clear()
        self.rev_combo.setCurrentIndex(0)
        self.operator_combo.setCurrentIndex(0)
        self.checked_combo.setCurrentIndex(0)
        self._apply_state(FORM)

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def closeEvent(self, event):
        self._stop_camera()
        event.accept()


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
