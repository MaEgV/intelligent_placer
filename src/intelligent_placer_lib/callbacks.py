import json

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

from src.intelligent_placer_lib.utils import transformed_sum


def get_save_img_callback(polygon, objects, path, prefix):
    cnt = 0

    def draw(xk, fk):
        nonlocal polygon, objects, prefix, cnt
        tmp = transformed_sum(polygon, objects, xk)
        tmp = cv.normalize(np.array(tmp, dtype=np.int32), None, 0, 255, cv.NORM_MINMAX).astype(int)
        if cnt == 0:
            masks = [polygon] + objects
            for i in range(len(objects) + 1):
                plt.imshow(masks[i])
                plt.savefig(f'{path}/detection/{i}.png', dpi=150)

        plt.imshow(tmp)
        plt.title(f"loss: {fk}")
        plt.savefig(f'{path}/{prefix}/{cnt}.png', dpi=150)
        cnt += 1

    return draw


def get_store_min_callback(on_update=None):
    m = [-1, -1]

    def store_min(xk, fk):
        nonlocal m
        if xk is None:
            return m
        elif m[1] == -1:
            m[0] = xk
            m[1] = fk
        elif m[1] > fk:
            print(f"UPDATED MIN {m[1]}=>{fk}")
            m[0] = xk
            m[1] = fk
            on_update and on_update(xk, fk)

    return store_min


def get_make_report_callback(objects, path, prefix, get_answer):
    def make_report(fk):
        nonlocal objects, prefix
        exp_name = path.split('/')[-2]
        answer = get_answer(fk)
        report = {
            "objects": {
                "computed": len(objects),
                "real": exp_name.split("_")[2]
            },
            "answer": {
                "computed": "y" if answer else "n",
                "real": exp_name.split("_")[1]
            },
            "metrics": {
                "loss": fk
            }
        }

        with open(f'{path}/{prefix}.json', 'w') as f:
            json.dump(report, f)

        return answer

    return make_report
