from lib.common import SETTINGS
import numpy as np
import torch

def box_convert(bboxes: list) -> list:
    """
    Format bounding boxes from [x,y,w,h] to [x1,y1,x2,y2]

    Arguments:
        bboxes (list): Bounding Boxes of shape (BATCH_SIZE, 4)
    
    Returns:
        list: Bounding boxes with the new format
    """
    new_boxes = []
    for box in bboxes:
        new_boxes.append([
            box[0] - box[3] / 2,
            box[1] - box[3] / 2,
            box[0] + box[3] / 2,
            box[1] + box[3] / 2
        ])

    return new_boxes


def iou(boxes_1: torch.Tensor, boxes_2: torch.Tensor) -> torch.Tensor:
    """
    Calculates intersection over union

    Arguments:
        boxes_1 (tensor): Bounding Boxes (BATCH_SIZE, 4)
        boxes_2 (tensor): Bounding Boxes (BATCH_SIZE, 4)

    Returns:
        tensor: Intersection over union for all examples
    """

    box1_x1 = boxes_1[..., 0:1] - boxes_1[..., 3:4] / 2
    box1_y1 = boxes_1[..., 1:2] - boxes_1[..., 3:4] / 2
    box1_x2 = boxes_1[..., 0:1] + boxes_1[..., 3:4] / 2
    box1_y2 = boxes_1[..., 1:2] + boxes_1[..., 3:4] / 2
    box2_x1 = boxes_2[..., 0:1] - boxes_2[..., 3:4] / 2
    box2_y1 = boxes_2[..., 1:2] - boxes_2[..., 3:4] / 2
    box2_x2 = boxes_2[..., 0:1] + boxes_2[..., 3:4] / 2
    box2_y2 = boxes_2[..., 1:2] + boxes_2[..., 3:4] / 2

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # Need clamp(0) in case they do not intersect, then we want intersection to be 0
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection)


def nms(
        bboxes: torch.Tensor,
        iou_threshold:float = SETTINGS.IOU_TRESHOLD,
        threshold:float = SETTINGS.CONFIDENCE_TRESHOLD,
        classes:list = None
    ) -> torch.Tensor:
    """
    Does Non Max Suppression given bboxes

    Arguments:
        bboxes (tensor)         : Bounding Boxes as format [x,y,w,h,score,class]. Shape: (BATCH_SIZE, 6)
        iou_threshold (float)   : threshold where predicted bboxes is correct
        threshold (float)       : threshold to remove predicted bboxes (independent of IoU)

    Returns:
        tensor: bboxes after performing NMS given a specific IoU threshold
    """

    bboxes = [box for box in bboxes if (1 / (1 + np.exp(-box[4]))) > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[4], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)
        bboxes = []
        for box in bboxes:
            iou_value = iou(torch.tensor(chosen_box[:4]),torch.tensor(box[:4]))
            #print(f"Box1: {chosen_box[:4]} - Box2: {box[:4]} - SALIDA: {iou_value}")
            if box[0] != chosen_box[0] and iou_value < iou_threshold:
                bboxes.append(box)

        if classes:
            if int(chosen_box[5]) not in classes: continue

        bboxes_after_nms.append(chosen_box)

    return np.array(bboxes_after_nms)