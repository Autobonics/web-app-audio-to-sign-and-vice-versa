import cv2
import numpy as np
import tkinter as tk
import speech_recognition as sr
from sp_proc import SpProc
from PIL import Image, ImageTk


class SpRecog:
    def __init__(self, window, title):
        self.window = window
        self.window.title(title)
        self.sp = SpProc()
        self.res_text = ""
        frame = np.zeros((self.sp.height, self.sp.width, 3), dtype=np.uint8)
        image = Image.fromarray(frame)
        self.image = ImageTk.PhotoImage(image)
        self.seq = None
        self.canvas = tk.Canvas(self.window, width=self.sp.width,
                                height=self.sp.height)
        self.canvas.pack()
        self.canvas.create_image(0, 0, anchor="nw", image=self.image)
        self.text_label = tk.Label(self.window, text="Res text : ")
        self.text_label.pack()
        self.text_box = tk.Text(self.window, height=10)
        self.text_box.pack()
        self.rec_start_btn = tk.Button(
            self.window, text="Record", command=self.rec_audio)
        self.rec_start_btn.pack()
        self.window.mainloop()

    def rec_audio(self):
        r = sr.Recognizer()
        with sr.Microphone() as source:
            r.adjust_for_ambient_noise(source)
            audio = r.listen(source)
            try:
                self.res_text = r.recognize_google(audio_data=audio)
                print("Res text : ", self.res_text)
                self.text_box.delete("1.0", tk.END)
                self.text_box.insert("1.0", chars=self.res_text)
                gloss = self.sp.get_gloss(self.res_text)
                self.seq = self.sp.get_seq(gloss, num=None)
                self.display_seq()
                if gloss:
                    print("Result gloss : ", gloss)
                else:
                    print("No matching gloss found")
            except Exception as err:
                print("Error Recognizing audio : ", err)

    def display_seq(self):
        if self.seq:
            for i, img in enumerate(self.seq):
                self.image = ImageTk.PhotoImage(image=Image.fromarray(img))
                self.canvas.create_image(0, 0, anchor="nw", image=self.image)
                self.window.update()
                if i == len(self.seq)-1:
                    break
                self.window.after(10)
        else:
            print("No sequence to display")


def main():
    root = tk.Tk()
    SpRecog(root, "Speech-Recog")


if __name__ == "__main__":
    main()
