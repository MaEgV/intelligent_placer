import glob
import json
import os

import cv2 as cv
import numpy as np
import skimage
import matplotlib.pyplot as plt


def shift(image, vector):
    return np.roll(image, vector, axis=(0, 1))


def transform(image, shifts, rotation):
    rotated = skimage.transform.rotate(image, rotation, order=0)
    return shift(rotated, shifts)


def transformed_sum(polygon, images, x):
    mask = np.zeros(polygon.shape) + polygon
    for i in range(len(images)):
        image = images[i]
        dx, dy, rotation = x[3*i:3*i+3]
        dx *= 8
        dy *= 8
        rotation *= 10
        mask += transform(np.array(image), [int(dx), int(dy)], rotation)
    return mask


def cacher(f):
    cache = {}
    cnt = 0

    def wrapper(*args, **kwargs):
        nonlocal cache, cnt, f
        hash = tuple(args[0])
        cached = cache.get(hash)
        if cached:
            print(f"RETURN CASHED VALUE-{cnt}")
            cnt += 1
            return cached

        res = f(*args, ** kwargs)
        cache[hash] = res

        return res

    return wrapper


def compress_image(src: np.ndarray, scale_percent: int):
    width = int(src.shape[1] * scale_percent / 100)
    height = int(src.shape[0] * scale_percent / 100)
    new_size = (width, height)

    return cv.resize(src, new_size)


def get_image_files(path: str) -> list:
    return glob.glob(os.path.join(path, r"*.jpg"))


def get_json_files(path: str) -> list:
    return glob.glob(os.path.join(path, r"*.json"))