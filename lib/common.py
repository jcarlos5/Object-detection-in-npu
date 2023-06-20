import json
import cv2

class SETTINGS:
    CONFIDENCE_TRESHOLD:float = 0.5
    IOU_TRESHOLD:float = 0.5
    IMAGE_SIZE:tuple = [640,640]
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
        #print(label)
        if isinstance(label, list):
            label = label[0]
        label = int(label)
        label = SETTINGS.CLASSES[label-1] if label <= len(SETTINGS.CLASSES) else f"CLASE: {label}"

        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        print(f"{label} of Type: {type(label)}")
        print(f"{score} of Type: {type(score)}")
        cv2.putText(image, '{0} {1:.2f}'.format(label, score[0]), (top, left - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)