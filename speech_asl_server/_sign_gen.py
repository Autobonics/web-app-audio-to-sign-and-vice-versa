import requests
import logging
import os
from pose_format import Pose
from pose_format.pose_visualizer import PoseVisualizer


class SignGen():
    def __init__(self, sentence):
        self.url_tmp = "https://spoken-to-signed-sxie2r74ua-uc.a.run.app/?slang={slang_val}&dlang={dlang_val}&sentence={sentence_val}"
        self.slang = "en"
        self.dlang = "us"
        self.sentence = sentence
        self.out_file = "out.gif"

    def req_pose(self) -> Pose:
        logging.info(f"Sending request for sentence : {self.sentence}")
        url = self.url_tmp.format(slang_val=self.slang,
                                  dlang_val=self.dlang, sentence_val=self.sentence)
        response = requests.request("GET", url)
        logging.info("Recieved response ")
        return Pose.read(response.content)

    def gen_feed(self) -> str:
        logging.info("ready to generate Gif")
        v = PoseVisualizer(self.req_pose())
        v.save_gif(self.out_file, v.draw())
        logging.info(f"Generated gif output for : '{self.sentence}'")
        return self.out_file

    def __del__(self):
        if os.path.isfile(self.out_file):
            try:
                os.remove(self.out_file)
            except Exception as err:
                logging.error(f"Unable to delete {self.out_file}\nErr : {err}")
