import cv2
from sp_proc import SpProc

sp = SpProc()
seq = sp.get_seq("HAPPY", 0)
print("len : ", len(seq))
print("writing images")
for i, img in enumerate(seq):
    cv2.imwrite(f'./test_img/{i}.png', img)
