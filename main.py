# from rknnlite.api import RKNNLite
from lib.common import SETTINGS
from lib.process import nms
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--inputtype", required=False, default="cam2",
	help="Select input cam, cam2, file")
ap.add_argument("-f", "--filename", required=False, default="skyfall.mp4",
	help="file video (.mp4)")
args = vars(ap.parse_args())


def draw(image, bboxes, scores, classes):
    """Draw the boxes on the image.

    Arguments:
        image: original image.
        bboxes (tensor) : Bounding boxes
        scores (tensor) : Scores of each class
        classes (list)  : Labels of bboxes
    """

    for box, score, label  in zip(bboxes, scores, classes):
        top, left, right, bottom = box
        label = SETTINGS.CLASSES[int(label)]

        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(label, score), (top, left - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        

if __name__ == '__main__':
    pass

    # rknn_lite = RKNNLite()
    # ret = rknn_lite.load_rknn(SETTINGS.MODEL_PATH)

    # if ret != 0: exit(ret)
    
    # ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0)

    # vs = open_cam_usb(config.CAM_DEV, config.CAM_WIDTH, config.CAM_HEIGHT)
