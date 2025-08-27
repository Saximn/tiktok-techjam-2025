import sys
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QListWidget, QComboBox, QMessageBox
)
from PyQt6.QtCore import QThread
import cv2
import pyvirtualcam

# Placeholder for enrolled faces
whitelist = []

# --- Video Thread for Webcam → OBS ---
class VideoThread(QThread):
    def __init__(self, width=640, height=480, fps=30):
        super().__init__()
        self.running = False
        self.width = width
        self.height = height
        self.fps = fps

    def run(self):
        cap = cv2.VideoCapture(0)
        with pyvirtualcam.Camera(width=self.width, height=self.height, fps=self.fps) as cam:
            self.running = True
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    break

                # TODO: send frame to your privacy model
                processed_frame = frame  # placeholder

                cam.send(processed_frame)
                cam.sleep_until_next_frame()
        cap.release()

    def stop(self):
        self.running = False
        self.wait()

# --- Main App ---
class PrivacyApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Privacy Streaming App")
        self.setGeometry(200, 200, 500, 500)

        self.streaming = False
        self.privacy_level = "Medium"
        self.video_thread = VideoThread()

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.init_whitelist_section()
        self.init_privacy_section()
        self.init_streaming_section()
        self.init_score_section()

    # --- Whitelist Section ---
    def init_whitelist_section(self):
        label = QLabel("Whitelist Enrollment")
        self.layout.addWidget(label)

        self.whitelist_list = QListWidget()
        self.layout.addWidget(self.whitelist_list)

        h_layout = QHBoxLayout()
        enroll_btn = QPushButton("Enroll Face")
        enroll_btn.clicked.connect(self.enroll_face)
        remove_btn = QPushButton("Remove Face")
        remove_btn.clicked.connect(self.remove_face)
        h_layout.addWidget(enroll_btn)
        h_layout.addWidget(remove_btn)
        self.layout.addLayout(h_layout)

    def enroll_face(self):
        # TODO: integrate face capture for real enrollment
        name = f"User {len(whitelist)+1}"
        whitelist.append(name)
        self.whitelist_list.addItem(name)
        QMessageBox.information(self, "Enroll Face", f"Enrolled {name} successfully!")

    def remove_face(self):
        selected_items = self.whitelist_list.selectedItems()
        if not selected_items:
            return
        for item in selected_items:
            whitelist.remove(item.text())
            self.whitelist_list.takeItem(self.whitelist_list.row(item))

    # --- Privacy Section ---
    def init_privacy_section(self):
        label = QLabel("Privacy Level")
        self.layout.addWidget(label)

        self.privacy_dropdown = QComboBox()
        self.privacy_dropdown.addItems(["Low", "Medium", "High"])
        self.privacy_dropdown.setCurrentText(self.privacy_level)
        self.privacy_dropdown.currentTextChanged.connect(self.change_privacy)
        self.layout.addWidget(self.privacy_dropdown)

    def change_privacy(self, level):
        self.privacy_level = level
        print(f"Privacy level set to {level}")
        # TODO: send level to your model

    # --- Streaming Section ---
    def init_streaming_section(self):
        label = QLabel("Streaming Control")
        self.layout.addWidget(label)

        h_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start Streaming")
        self.start_btn.clicked.connect(self.start_streaming)
        self.stop_btn = QPushButton("Stop Streaming")
        self.stop_btn.clicked.connect(self.stop_streaming)
        self.stop_btn.setEnabled(False)
        h_layout.addWidget(self.start_btn)
        h_layout.addWidget(self.stop_btn)
        self.layout.addLayout(h_layout)

        self.status_label = QLabel("Status: Stopped")
        self.layout.addWidget(self.status_label)

    def start_streaming(self):
        self.video_thread.start()
        self.streaming = True
        self.status_label.setText("Status: Streaming")
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        print("Streaming started")

    def stop_streaming(self):
        self.video_thread.stop()
        self.streaming = False
        self.status_label.setText("Status: Stopped")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        print("Streaming stopped")
        self.show_score()

    # --- Post-Stream Score Section ---
    def init_score_section(self):
        self.score_label = QLabel("Post-Stream Score: N/A")
        self.layout.addWidget(self.score_label)

    def show_score(self):
        # TODO: get score from model
        score = 87  # example
        self.score_label.setText(f"Post-Stream Score: {score}")

# --- Run App ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PrivacyApp()
    window.show()
    sys.exit(app.exec())
