import os
import random
import cv2
import numpy as np
from typing import List, Optional
from gloss_proc import GlossProcess, draw_landmarks, get_vid_data
from .sp_utils import lm_mp


class SpProc:
    def __init__(self):
        self.gp = GlossProcess.load_checkpoint()
        self.glosses = set(self.gp.glosses)
        self.gloss_dir = self.gp.gloss_dir
        self.vid_cnt = self.gp.vid_count
        self.frame_cnt = self.gp.frame_count
        cap = cv2.VideoCapture(0)
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def get_seq(self, gloss: str, num: Optional[int]) -> List[np.ndarray]:
        gloss = self.gp._sanitize([gloss])[0]
        if gloss in self.glosses:
            vid = self.get_vid(gloss, num)
            return [self.get_frame(frame_arr) for frame_arr in vid]
        else:
            raise Exception(f"Gloss {gloss} not Found!")

    def get_vid(self, gloss: str, num: Optional[int]) -> np.ndarray:
        if num and num < self.vid_cnt:
            vid_id = num
        else:
            vid_id = random.randint(0, self.vid_cnt-1)
        vid_path = f"./{self.gloss_dir}/{gloss}/{vid_id}.csv"
        print(f"Getting vid : {vid_path}")
        return get_vid_data(vid_path)

    def get_frame(self, lm_arr: np.ndarray) -> np.ndarray:
        frame = np.full((self.height, self.width, 3), 255, dtype=np.uint8)
        lm = lm_mp(lm_arr)
        return draw_landmarks(frame, lm)
