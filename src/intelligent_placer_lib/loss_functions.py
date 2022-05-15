import cv2 as cv
import numpy as np

from src.intelligent_placer_lib.utils import transformed_sum, cacher


def get_loss_func(polygon, images, out_c=150, intersection_c=4, callbacks=None):
    # @cacher
    def loss_func(x: tuple, Y=0):
        nonlocal polygon, images, out_c, intersection_c

        tmp = transformed_sum(polygon, images, x)
        inside = tmp[polygon.astype('uint8') != 0].flatten() // 255
        outside = tmp[polygon.astype('uint8') == 0] // 255
        # outside = cv.bitwise_and(tmp.astype('uint8'), cv.bitwise_not(polygon.astype('uint8'))).flatten() // 255
        inside_n = inside[inside > 2] ** 2
        outside_n = outside * out_c
        res = sum(inside_n) + sum(outside_n)
        for callback in callbacks:
            callback(x, res)
        callbacks and map(lambda cb: cb(x, res), callbacks)
        return res

    return loss_func
