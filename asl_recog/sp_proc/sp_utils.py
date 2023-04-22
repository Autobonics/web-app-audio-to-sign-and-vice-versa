from enum import Enum
from typing import Dict, List
import numpy as np


Landmark = List[Dict[str, float]]
LandmarkDict = Dict[str, Landmark]


class LmType(Enum):
    FaceLm = (468, 3),
    HandLm = (21, 3),
    PoseLm = (33, 2),


def lm_from_np(arr: np.ndarray, lm_type=LmType) -> LandmarkDict:
    res_list: List[List[float]] = arr.reshape(lm_type.value).tolist()
    if lm_type != LmType.PoseLm:
        res_lm = list(map(
            lambda arr: {'x': arr[0], 'y': arr[1], 'z': arr[2]}, res_list))
    else:
        res_lm = list(map(
            lambda arr: {'x': arr[0], 'y': arr[1], 'z': 0}, res_list))
    return {'landmark': res_lm}


def lm_all_np(arr: np.ndarray) -> Dict[str, LandmarkDict]:
    face_lm = lm_from_np(arr[:1404], lm_type=LmType.FaceLm)
    rt_hand_lm = lm_from_np(arr[1404:1467], lm_type=LmType.HandLm)
    lf_hand_lm = lm_from_np(arr[1467:1530], lm_type=LmType.HandLm)
    pose_lm = lm_from_np(arr[1530:1596], lm_type=LmType.PoseLm)
    return {
        'right_hand_landmarks': rt_hand_lm,
        "left_hand_landmarks": lf_hand_lm,
        "pose_landmarks": pose_lm,
        "face_landmarks": face_lm,
    }
