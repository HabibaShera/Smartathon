import time
import cv2
import numpy as np


def cal_time(fun): # a simple wraper around a function to calculate the runtime
    def wraper(*args, **kwargs):
        st = time.time()
        ret = fun(*args, **kwargs)
        fn = time.time()
        return ret, (fn - st)

    return wraper


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    """
    :param img:array the image you want to reshape
    :param new_shape: tuple or list the new_shape
    :param color:tuple color of the padding
    :param auto:bool auto reshaping
    :param scaleFill: bool whether to stretch the padding
    :param scaleup:bool whether the padding or resizing should be down or up
    :param stride: int determine the stride for the resizing
    :return: a resized image
    """
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def load_img(img_path, form='CHW', **kwargs):
    """
    :param img_path:str the specified path to the image
    :param form:str the form of training or inference if form="CHW" else normal loading of the image
    :param kwargs:kwargs for the resizing method
    :return: loaded_img as a numpy array
    """
    img = cv2.imread(img_path)
    img = img[..., ::-1]
    shape = img.shape
    if form == 'CHW':
        img, _, _ = letterbox(img, **kwargs)
        img = img.transpose(2, 0, 1)  # transposing from (h,w,c) to (c,h,w) for training
    return np.ascontiguousarray(img), shape
