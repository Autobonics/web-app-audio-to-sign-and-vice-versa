import os
import cv2
import sys
import pickle
import torch
import pyttsx3
import numpy as np
import pandas as pd
import mediapipe as mp
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer
from typing import Tuple, Union, List
from gloss_proc import proc_landmarks, Landmarks, GlossProcess, draw_landmarks
from sp_proc import SpProc
from sp_proc.sp_utils import lm_mp
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


class VidProcess:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def get_frame(self) -> Tuple[bool, Union[np.ndarray, None]]:
        success, image = False, None
        if self.cap.isOpened():
            success, image = self.cap.read()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return success, image

    def draw_lm(self, image: np.ndarray, res: Landmarks) -> np.ndarray:
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return draw_landmarks(image, res)

    def get_lm(self, image: np.ndarray) -> Union[Landmarks, None]:
        image.flags.writeable = False
        with mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as holistic:
            try:
                return holistic.process(image)
            except:
                return None

    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()


class AslRecogApp(QtWidgets.QMainWindow):
    def __init__(self, parent=None, max_seq_len: int = 24, model: str = "asl-recog_lstm.pt"):
        super(AslRecogApp, self).__init__(parent)
        self.vid_proc = VidProcess()
        self.tts = pyttsx3.init()
        self.tts.setProperty('rate', 125)
        self.max_seq_len = max_seq_len
        self.image = None
        self.capturing = False
        self.seq: List[np.ndarray] = []
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.load(model)
        self.gp = GlossProcess.load_checkpoint()
        self.classes = self.gp.glosses
        self.res_text: str = ""

        self.setWindowTitle("AslRecog")
        self.setFixedSize(800, 600)

        self.central_widget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.central_layout = QtWidgets.QVBoxLayout(self.central_widget)

        self.show_landing_page()

        self.image_label = QtWidgets.QLabel(self)
        self.central_layout.addWidget(self.image_label)

        self.start_button = QtWidgets.QPushButton(self)
        self.start_button.setText("Start Capturing")
        self.start_button.clicked.connect(self.start_capturing)
        self.central_layout.addWidget(self.start_button)

        self.text_box = QtWidgets.QPlainTextEdit(self)
        self.text_box.setReadOnly(True)
        self.central_layout.addWidget(self.text_box)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(1)

    def show_landing_page(self):
        image_path = "asl-recog.png"
        self.central_widget.setStyleSheet(
            f"background-image: url({image_path});")

    def start_capturing(self):
        self.capturing = True
        self.start_button.hide()

    def update(self):
        if self.capturing:
            success, frame = self.vid_proc.get_frame()
            if not success:
                return
            if (len(self.seq)+1 == self.max_seq_len):
                res = self.vid_proc.get_lm(frame)
                if res:
                    self.seq.append(proc_landmarks(res))
                    proc_seq = torch.from_numpy(
                        np.array(self.seq)).float().to(self.device)
                    self.model.eval()
                    with torch.no_grad():
                        x = proc_seq.unsqueeze(0)
                        out = self.model(x)
                    res_class = torch.argmax(out, dim=1)
                    self.tts.say(self.classes[res_class.item()])
                    self.tts.runAndWait()
                    self.tts.stop()
                    self.res_text += self.classes[res_class.item()]+".\n"
                    self.text_box.clear()
                    self.text_box.insertPlainText(self.res_text)
                    self.seq = []
                    if len(self.res_text.splitlines()) > 5:
                        self.res_text = ""
            res = self.vid_proc.get_lm(frame)
            if res:
                self.seq.append(proc_landmarks(res))
            frame = frame if not res else self.vid_proc.draw_lm(frame, res)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, channel = frame.shape
            bytesPerLine = 3 * width
            self.image = QImage(frame.data, width, height,
                                bytesPerLine, QImage.Format_RGB888)
            self.image_label.setPixmap(QPixmap.fromImage(self.image))
            self.image_label.setScaledContents(True)


def main():
    app = QtWidgets.QApplication(sys.argv)
    asl_recog_app = AslRecogApp()
    asl_recog_app.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
