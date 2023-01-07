import torchvision
import torch
import numpy as np


def xywh2xyxy(bboxes):
    """
    :param bboxes: tensor or numpy array of shape(bs,4) format (xc,yc,w,h)
    :return: transformed bbox converted to format (xmin,ymin,xmax,ymax)
    """
    y = bboxes.clone() if isinstance(bboxes, torch.Tensor) else np.copy(bboxes)
    y[:, :2] = bboxes[..., :2] - bboxes[..., 2:] / 2
    y[:, 2:] = bboxes[..., :2] + bboxes[..., 2:] / 2
    return y


def NMS(preds, iou_threshold, prob_threshold, multi_label=False, max_boxes=100, agnostic=False):
    """
    :param preds:list(numpy arrays or torch tensors)
    :param iou_threshold:float threshold for predicting the best bounding box
    :param prob_threshold: float probability score for predicting bounding box
    :param multi_label:bool to determine whether to take the all labels in the image
    :param max_boxes:int to determine the number of boxes before nms
    :param agnostic:bool for agnostic predictions
    :return: best_bounding boxes
    """
    max_wh = 4096
    confs = preds[..., 4] > prob_threshold
    num_classes = preds.shape[2] - 5
    outs = [torch.zeros((0, 6)) for _ in range(preds.shape[0])]

    for img_index, img_pred in enumerate(preds):
        img_pred = img_pred[confs[img_index]]
        if not img_pred.shape[0]:
            break
        if num_classes == 1:
            img_pred[:, 5:] = img_pred[:, 4:5]
        else:
            img_pred[:, 5:] *= img_pred[:, 4:5]

        box_pred = xywh2xyxy(img_pred[:, :4])
        if multi_label:
            i, j = (img_pred[:, 5:] > prob_threshold).nonzero(as_tuple=False).T
            img_pred = torch.cat((box_pred[i], img_pred[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = img_pred[:, 5:].max(1, keepdim=True)
            img_pred = torch.cat((box_pred, conf, j.float()), 1)[conf.view(-1) > prob_threshold]
        n_examples = img_pred.shape[0]
        if not n_examples:
            continue
        elif n_examples > max_boxes:
            img_pred = img_pred[img_pred[:, 4].argsort(descending=True)[:max_boxes]]
        c = img_pred[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = img_pred[:, :4] + c, img_pred[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_threshold)  # NMS
        outs[img_index] = img_pred[i]
    return outs


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    """
    :param img1_shape:tuple or list the resized shape of the image
    :param coords:numpy array or tensor the bbox
    :param img0_shape:tuple or list the original shape of the image
    :param ratio_pad: float the pad_ratio
    :return: scaled bbox for the original image
    """
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def clip_coords(boxes, img_shape):
    """
    :param boxes:numpy array or tensor  the bbox
    :param img_shape: tuple or list the shape you want to clip on
    :return: clipped bbox
    """
    boxes[:, 0].clamp_(0, img_shape[1])
    boxes[:, 1].clamp_(0, img_shape[0])
    boxes[:, 2].clamp_(0, img_shape[1])
    boxes[:, 3].clamp_(0, img_shape[0])
