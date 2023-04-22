import cv2
import pickle
import tkinter
import numpy as np
import pandas as pd
import mediapipe as mp
import torch
from PIL import Image, ImageTk
from typing import Tuple, Union, List
from gloss_proc import proc_landmarks, Landmarks, GlossProcess
from gloss_proc.utils import draw_landmarks

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


class AslRecogApp:
    def __init__(self, window, title, max_seq_len: int = 24, model: str = "asl-recog_lstm.pt"):
        self.window = window
        self.vid_proc = VidProcess()
        self.window.title(title)
        self.max_seq_len = max_seq_len
        self.image = ImageTk.PhotoImage(file="asl-recog.png")
        self.seq: List[np.ndarray] = []
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.load(model)
        self.gp = GlossProcess.load_checkpoint()
        self.classes = self.gp.glosses
        self.res_text: str = ""
        self.canvas = tkinter.Canvas(
            self.window, width=self.vid_proc.width, height=self.vid_proc.height)
        self.canvas.pack()
        self.text_label = tkinter.Label(self.window, text="Res text : ")
        self.text_label.pack()
        self.text_box = tkinter.Text(self.window, height=10)
        self.text_box.pack()
        self.update()
        self.window.mainloop()

    def update(self):
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
                    out = self.model(proc_seq)
                res_class = torch.argmax(torch.max(out, dim=0).values)
                self.res_text += self.classes[res_class.item()]+".\n"
                self.text_box.delete("1.0", tkinter.END)
                self.text_box.insert("1.0", chars=self.res_text)
                self.seq = []
                if len(self.res_text.splitlines()) > 5:
                    self.res_text = ""
        res = self.vid_proc.get_lm(frame)
        if res:
            self.seq.append(proc_landmarks(res))
        frame = frame if not res else self.vid_proc.draw_lm(frame, res)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.image = ImageTk.PhotoImage(image=Image.fromarray(frame))
        self.canvas.create_image(0, 0, image=self.image, anchor=tkinter.NW)
        self.window.after(1, self.update)


def main():
    root = tkinter.Tk()
    AslRecogApp(root, "AslRecog")


if __name__ == "__main__":
    main()
