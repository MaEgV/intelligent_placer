import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from src.intelligent_placer_lib.object import Object
from src.intelligent_placer_lib.utils import get_polygon


class Polygon:
    """
    Implementation of a polygon located in a picture
    Calculates its own contour by itself
    Provide possibility to place objects of the OBject class inside yourself
    """
    def __init__(self, image: np.ndarray):
        self._image = image
        self._contour = get_polygon(image)
        self._placed = []
        self._area = None

    def get_area(self):
        if self._area is None:
            self._area = cv.contourArea(self._contour)
        return cv.contourArea(self._contour)

    def get_image(self):
        return self._image

    def _lay_object(self, obj: Object) -> bool:
        # Try to find free space inside Polygon
        res = sum(list(map(lambda x: x.get_area(), self._placed))) + obj.get_area() < self.get_area()
        print(f'obj: {obj._id}, state: {res}, already placed: {[str(i) for i in self._placed]}')
        return res

    def place_object(self, obj: Object) -> bool:
        # Try to insert new Object inside Polygon
        if self._lay_object(obj):
            self._placed.append(obj)
            return True
        return False

    def save_image(self, path: str, prefix: str):
        image_with_contour = cv.drawContours(self.get_image(), [self._contour], -1, (255, 0, 0), 3, cv.LINE_AA)
        plt.imshow(image_with_contour)
        plt.savefig(f'{path}{prefix}', dpi=150)