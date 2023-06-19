import json
import cv2

class SETTINGS:
    CONFIDENCE_TRESHOLD:float = 0.5
    IOU_TRESHOLD:float = 0.5
    IMAGE_SIZE:tuple = (640,640)
    CLASSES:list = ["PERSON"]
    MODEL_PATH:str = "./models/model.rknn"
    CAM_DEV:int = 0
    CAM_WITH:int = 1280
    CAM_HEIGHT:int = 720

    @staticmethod
    def load(fname:str="../config.json"):
        with open(fname, 'r') as fp:
            for k, v in json.load(fp).items():
                setattr(SETTINGS, k, v)


def open_cam(type:int = 0, filename:str = None):
    """
    Initialize video stream capture
    Arguments:
        type(int): Type of capture

    Returns:
        VideoCapture
    """
    if type == 0:
        vs = cv2.VideoCapture(0)
    elif type == 1:
        vs = cv2.VideoCapture(filename)
    
    vs.set(cv2.CAP_PROP_FRAME_WIDTH, SETTINGS.CAM_WITH)
    vs.set(cv2.CAP_PROP_FRAME_HEIGHT, SETTINGS.CAM_HEIGHT)

    return vs


def letterbox(im, new_shape:tuple=tuple(SETTINGS.IMAGE_SIZE), color:tuple=(0, 0, 0)):
    """
    Resize and pad image while meeting stride-multiple constraints
    """

    shape = im.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return im, ratio, (dw, dh)

def draw(image, bboxes, scores, classes):
    """Draw the boxes on the image.

    Arguments:
        image: original image.
        bboxes (tensor) : Bounding boxes
        scores (tensor) : Scores of each class
        classes (list)  : Labels of bboxes
    """

    for box, score, label  in zip(bboxes, scores, classes):
        top, left, right, bottom = [int(n) for n in box]
        label = int(label)
        label = SETTINGS.CLASSES[label-1] if label <= len(SETTINGS.CLASSES) else f"CLASE: {label}"

        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(label, score), (top, left - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)