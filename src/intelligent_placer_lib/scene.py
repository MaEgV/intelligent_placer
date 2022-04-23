import numpy as np
import cv2 as cv

from src.intelligent_placer_lib.object import Object
from src.intelligent_placer_lib.polygon import Polygon
from pylab import array, arange, uint8
import matplotlib.pyplot as plt


class Scene:
    """
    General class for images. Contains Polygon and Objects
    This class implements the logic of interaction between a Polygon and Objects
    It can find the specified Objects in at input image and transforms them
    """
    def __init__(self, image: np.ndarray, all_objects: list, path:str = None):
        self._image = image
        self._objects = self.get_objects_in_scene(all_objects)
        self._polygon = Polygon(image[0:int(len(image) * 0.65)])
        if path:
            self._polygon.save_image(path, path.split('/')[-1])

    def transform(self, image):
        # Prepare images
        maxIntensity = 255.0  # depends on dtype of image data
        x = arange(maxIntensity)
        phi = 1
        theta = 1
        y = (maxIntensity / phi) * (x / (maxIntensity / theta)) ** 0.5
        newImage1 = (maxIntensity / phi) * (image / (maxIntensity / theta)) ** 2
        newImage1 = array(newImage1, dtype=uint8)
        return newImage1

    def get_matches_and_transform(self, obj: Object, min_match=10, accuracy=0.8) -> tuple:
        # Find key points and create transform function for contours
        sift = cv.SIFT_create()
        # find the keypoints and descriptors with SIFT
        im1 = self.transform(obj.get_image())
        im2 = self.transform(self._image[int(len(self._image)*0.5):])

        kp1, des1 = sift.detectAndCompute(im1.astype('uint8'), None)
        kp2, des2 = sift.detectAndCompute(im2.astype('uint8'), None)
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)

        # store all the good matches as per Lowe's ratio test.
        good = []
        for m, n in matches:
            if m.distance < accuracy * n.distance:
                good.append(m)

        if len(good) <= max(obj.kp_count/8, 8):
            print(f"Did not found object: {obj} at scene. Mathces: {len(good)}")
            return False, lambda x: x

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
        transform = lambda contour: cv.perspectiveTransform(np.float32(contour), M)

        # Draw matches
        h, w, _ = im1.shape
        pts = np.float32(obj.get_contour()).reshape(-1, 1, 2)
        try:
            dst = cv.perspectiveTransform(pts, M)
        except:
            return False, lambda x: x
        img2 = cv.polylines(im2, [np.int32(dst)], True, 255, 3, cv.LINE_AA)
        im1 = cv.drawContours(im1, [obj._contour], -1, (255, 0, 0), 3, cv.LINE_AA)
        draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)
        img3 = cv.drawMatches(im1, kp1, img2, kp2, good, None, **draw_params)
        plt.imshow(img3, 'gray')
        plt.savefig(f'data/objects/matches/{obj._id}.png')

        print(f"Found object: {obj} at scene. Mathces: {len(good)}")

        return True, transform

    def get_objects_in_scene(self, objects: list) -> list:
        objects_in_scene = []
        for obj in objects:
            is_found, transform = self.get_matches_and_transform(obj)
            if is_found and obj.check_state():
                obj.transform(transform)
                objects_in_scene.append(obj)
        return objects_in_scene

    def place_objects(self):
        for i in self._objects:
            if not self._polygon.place_object(i):
                return False
        return True
    