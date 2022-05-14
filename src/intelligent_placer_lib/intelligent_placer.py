import cv2 as cv
from src.intelligent_placer_lib.scene import Scene
from src.intelligent_placer_lib.utils import fit, load_objects


def check_image(image, is_fitted=False, objects_path="data/objects/"):
    if not is_fitted:
        print(f"Process objects")
        fit(objects_path)
        print(f"Objects successfully processed")

    img = cv.imread(image)
    objects = load_objects(objects_path)
    scene = Scene(img, objects, objects_path)
    return scene.place_objects()
