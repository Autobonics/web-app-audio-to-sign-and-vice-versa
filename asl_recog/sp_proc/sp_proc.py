from typing import List
import numpy as np
from gloss_proc import GlossProcess


class SpProc:
    def __init__(self):
        gp = GlossProcess.load_checkpoint()
        self.glosses = set(gp.glosses)

    def get_seq(self, gloss: str) -> List[np.ndarray]:
        return []
