import cv2 as cv
import json
import numpy as np
import matplotlib.pyplot as plt


class Object:
    """
    Implementation of an object located at image with a known contour
    Allows you to calculate the area and transform the contour and use files to store data
    """
    def __init__(self, id, image: np.ndarray, contour: np.ndarray, kp_count: int):
        self._id = id
        self._image = image
        self._contour = contour
        self.kp_count = kp_count  # Count of key points. Needs as a parameter of the classification
        self._area = None

    def check_state(self):
        # Validate state of the object. If False, cant use some methods
        return type(self._contour) == np.ndarray and len(self._contour) > 0

    def transform(self, transform: callable) -> None:
        # Perform geometric transformation of the self._contour
        self._contour = transform(self._contour)
        self._area = None

    def get_image(self) -> np.ndarray:
        return self._image

    def get_area(self):
        if self._area is None:
            self._area = cv.contourArea(self._contour)
        return self._area

    def get_contour(self):
        return self._contour

    def to_dict(self):
        return {"id": self._id, "contour": self._contour.tolist(), "kp_count": self.kp_count}

    def save_to_json(self, path):
        with open(f'{path}/{self._id}.json', 'w') as f:
            json.dump(self.to_dict(), f)

    def save_image(self, path: str):
        if self.check_state():
            image_with_contour = cv.drawContours(self.get_image(), [self._contour], -1, (255, 0, 0), 3, cv.LINE_AA)
            plt.imshow(image_with_contour)
            plt.savefig(f'{path}{self._id}.png', dpi=150)

    def __str__(self):
        return f"[id: {self._id}, state: {self.check_state()}]"


class ObjectCreator:
    @staticmethod
    def from_dict(input_dict: dict) -> Object:
        return Object(**input_dict)

    @staticmethod
    def from_file(path_to_json: str, image: np.ndarray) -> Object:
        with open(path_to_json, 'r') as f:
            input_dict = json.load(f)
        input_dict['image'] = image[int(len(image)*0.1):int(len(image)*0.9), int(len(image[0])*0.1):int(len(image[0])*0.9)]
        if len(input_dict['contour']) > 0:
            input_dict['contour'] = np.array(input_dict['contour'])
        return ObjectCreator.from_dict(input_dict)


