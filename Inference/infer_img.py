from ..utils import detection_utils as dt
from ..utils import model_utils
from ..utils import Images_utils
import torch
import numpy as np
from typing import Union, List, Tuple
import dataclasses


class YoloOutputs(dataclasses):
    bboxes: np.array = None
    score: Union[List[int], Tuple[int]] = None
    Class: Union[List[int], Tuple[int]] = None


def infer_img(model: Union[model_utils.TracedModel, torch.nn.Module],
              path=None, iou_th=0.4,
              score_th=0.25, **kwargs):
    assert path is None, "can't infer an image that doesn't exist"
    img, shape = Images_utils.load_img(path, form='CHW', **kwargs)
    img = torch.from_numpy(img) / 255.0
    img = img.unsqueeze(0)
    with torch.no_grad():
        pred = model(img)[0]
    pred = dt.NMS(pred, iou_th, score_th)
    pred = pred.cpu().numpy()
    predictions = []
    for i, det in enumerate(pred):
        det[:, :4] = dt.scale_coords(img.shape[1:], det[:, :4], shape)
        scores = det[:, 4].tolist()
        classes = det[:, 5].tolist()
        predictions.append(YoloOutputs(bboxes=det[:, :4], scores=scores, Class=classes))
    return predictions
