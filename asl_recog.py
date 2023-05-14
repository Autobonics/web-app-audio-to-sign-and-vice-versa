import cv2
import pickle
import torch
import pyttsx3
import tkinter as tk
import numpy as np
import pandas as pd
import mediapipe as mp
from tkinter import ttk
from PIL import Image, ImageTk
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


class AslRecogApp:
    def __init__(self, window, title, max_seq_len: int = 24, model: str = "asl-recog_lstm.pt"):
        self.window = window
        self.vid_proc = VidProcess()
        self.tts = pyttsx3.init()
        self.tts.setProperty('rate', 125)
        self.window.title(title)
        self.max_seq_len = max_seq_len
        self.image = ImageTk.PhotoImage(file="asl-recog.png")
        self.capturing = False
        self.seq: List[np.ndarray] = []
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.load(model)
        self.gp = GlossProcess.load_checkpoint()
        self.classes = self.gp.glosses
        self.res_text: str = ""
        self.canvas = tk.Canvas(
            self.window, width=self.vid_proc.width, height=self.vid_proc.height)
        self.text_label = ttk.Label(self.window, text="Res text : ")
        self.text_box = tk.Text(self.window, height=10)

        # Landing Page
        self.landing_page = tk.Frame(
            self.window, width=self.vid_proc.width, height=self.vid_proc.height)
        self.landing_page.pack()
        landing_image = Image.open("asl-recog.png")
        landing_image = landing_image.resize((500, 500), Image.ANTIALIAS)
        landing_image = ImageTk.PhotoImage(landing_image)
        landing_label = ttk.Label(self.landing_page)
        landing_label.configure(image=landing_image)
        landing_label.pack(pady=50)
        start_button = ttk.Button(
            self.landing_page, text="Start Capturing", command=self.start_capturing)
        start_button.pack(pady=20)

        self.update()
        self.window.mainloop()

    def start_capturing(self):
        self.capturing = True
        self.landing_page.pack_forget()
        self.canvas.pack()
        self.text_label.pack()
        self.text_box.pack()
        self.update()

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
                    self.text_box.delete("1.0", tk.END)
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
            self.canvas.create_image(0, 0, image=self.image, anchor=tk.NW)
            self.window.after(1, self.update)


def main():
    root = tk.Tk()
    style = ttk.Style()
    style.configure("TLabel", font=("Arial", 14), foreground="blue")
    style.configure("TText", font=("Arial", 12), background="lightgray")
    AslRecogApp(root, "AslRecog")


if __name__ == "__main__":
    main()
