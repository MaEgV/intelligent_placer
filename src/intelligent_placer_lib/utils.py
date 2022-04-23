import glob
import json
import os

import cv2 as cv
import numpy as np
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from skimage.feature import sift

from src.intelligent_placer_lib.object import ObjectCreator


def get_image_files(path: str) -> list:
    return glob.glob(os.path.join(path, r"*.jpg"))


def get_json_files(path: str) -> list:
    return glob.glob(os.path.join(path, r"*.json"))


def load_objects(path: str) -> list:
    objects = []
    for file in get_image_files(path):
        id = os.path.splitext(os.path.basename(file))[0]
        image = cv.imread(file)
        objects.append(ObjectCreator.from_file(file.replace('jpg', 'json'), image))
    return objects


def get_polygon(image):
    # Grayscale
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # Find Canny edges
    edged = cv.Canny(gray, 220, 220)
    # Finding Contours
    contours, hierarchy = cv.findContours(edged, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    max_contour = max(contours, key=lambda x: len(x))
    return max_contour


def find_contours(image):
    image = image.copy()
    canny_image = cv.Canny(image, 40, 250)  # empirical parameters!!!

    # applying closing function
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (7, 7))
    closed = cv.morphologyEx(canny_image, cv.MORPH_CLOSE, kernel)

    # finding_contours
    (contours, _) = cv.findContours(closed.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    return contours


def get_object_contour(image):
    contours = find_contours(image)
    if len(contours) < 2:
        return np.array([])
    # return sorted(contours, key=lambda x: -cv.contourArea(x))[1]
    return max(contours, key=lambda x: cv.contourArea(x))


def get_special_points(image, filename=None):
    image = np.copy(image)
    sift = cv.SIFT_create()
    kp1, des1 = sift.detectAndCompute(image.astype('uint8'), None)
    for kp in kp1:
        image = cv.circle(image, list(map(int, kp.pt)), radius=5, color=(55, 125, 160), thickness=-1)

    if filename:
        plt.imshow(image)
        plt.savefig(filename, dpi=150)

    return kp1


def create_object_json(path, image, id):
    cropped_image = image[int(len(image)*0.1):int(len(image)*0.9), int(len(image[0])*0.1):int(len(image[0])*0.9)]
    obj_contour = get_object_contour(cropped_image)
    obj_points = get_special_points(cropped_image, f'{path}/key_points/{id}.png')
    obj_dict = {'id': id, 'contour': obj_contour, 'image': cropped_image, 'kp_count': len(obj_points)}
    obj = ObjectCreator.from_dict(obj_dict)
    obj.save_to_json(path)
    obj.save_image(path+'/fitted/')


def fit(path):
    for file in get_image_files(path):
        id = os.path.splitext(os.path.basename(file))[0]
        image = cv.imread(file)
        create_object_json(path, image, id)
