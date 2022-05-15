import os

import numpy as np
from imageio import imread
from src.intelligent_placer_lib.callbacks import get_save_img_callback, get_store_min_callback, get_make_report_callback
from src.intelligent_placer_lib.detector import get_masks
from src.intelligent_placer_lib.loss_functions import get_loss_func
from src.intelligent_placer_lib.optimizer import optimize
from src.intelligent_placer_lib.utils import compress_image


def check_image(image, path=None):
    compressed_image = compress_image(imread(image), 70)
    polygon_mask, object_masks = get_masks(compressed_image)
    print(image)
    callbacks = []
    postprocess = lambda fk: fk < 1300 * len(object_masks)
    if path:
        report_path = path + image.split('\\')[-1].split('.')[0] + '/'
        detections_path = report_path + '/detection/'
        loss_path = report_path + '/loss/'

        if not os.path.exists(report_path):
            os.makedirs(report_path)

        if not os.path.exists(detections_path):
            os.makedirs(detections_path)

        if not os.path.exists(loss_path):
            os.makedirs(loss_path)

        callbacks.append(get_store_min_callback(get_save_img_callback(polygon_mask, object_masks, report_path, "loss")))
        postprocess = get_make_report_callback(object_masks, report_path, 'report', postprocess)

    loss = get_loss_func(polygon_mask, object_masks, callbacks=callbacks)

    bounds = [(-15, 15)] * 3 * len(object_masks)

    ans = postprocess(optimize(loss, bounds))

    return ans
