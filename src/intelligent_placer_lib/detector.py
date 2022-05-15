import cv2 as cv
import numpy as np
from scipy.ndimage.morphology import binary_fill_holes
from skimage.color import rgb2gray
from matplotlib import pyplot as plt
from skimage.segmentation import watershed
from skimage.morphology import binary_closing, binary_erosion
from skimage.feature import canny


def find_contours_canny(image, s=2.2, mask=None):
    canny_image = canny(image, sigma=s).astype('uint8')

    h, t = int(len(canny_image) * 0.03), int(len(canny_image[0]) * 0.03)
    canny_image = binary_closing(canny(image, sigma=s), selem=np.ones((14, 14)))
    canny_image = canny_image[h:int(len(canny_image) * 0.97), t:int(len(canny_image[0]) * 0.97)]
    canny_image = binary_fill_holes(canny_image)
    plt.imshow(canny_image)
    (contours, _) = cv.findContours(canny_image.astype('uint8'), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for c in contours:
        c += [h, t]
    return contours


def shift_cnt_to_center(contour, shape):
    M = cv.moments(contour)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return contour - [cX, cY] + [shape[1] // 2, shape[0] // 2]


def get_polygon_mask(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)[:int(len(image)*0.75)]
    edged = cv.Canny(gray, 220, 220)

    contours, hierarchy = cv.findContours(edged, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    max_contour = max(contours, key=lambda x: len(x))
    centered_contour = shift_cnt_to_center(max_contour, image.shape)

    mask = np.zeros(image.shape)
    mask = cv.drawContours(mask, np.array([centered_contour]), -1, (255, 0, 0), -1, cv.LINE_AA).astype(int)

    return mask


def get_object_masks(image):
    image_slice = int(len(image) * 0.5)
    contours = []
    contours += find_contours_canny(rgb2gray(image[image_slice:]), mask=[])

    masks = []
    for contour in contours:
        if cv.contourArea(contour) < 200:
            continue
        centered_contour = shift_cnt_to_center(np.array(contour), image.shape)
        mask = cv.drawContours(np.zeros(image.shape), np.array([centered_contour]), -1, (255, 0, 0), -1,
                               cv.LINE_AA).astype(int)
        masks.append(mask)

    return masks


def get_masks(img):
    polygon_mask = get_polygon_mask(img)
    object_masks = get_object_masks(img)
    return polygon_mask, object_masks
