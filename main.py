# from rknnlite.api import RKNNLite
from lib.common import SETTINGS, open_cam_usb, letterbox
from imutils.video import FPS
from lib.process import nms
import argparse
import time
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
    # rknn_lite = RKNNLite()
    # ret = rknn_lite.load_rknn(SETTINGS.MODEL_PATH)

    # if ret != 0: exit(ret)
    
    # ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0)

    vs = open_cam_usb()
    time.sleep(2.0)
    fps = FPS().start()

    if not vs.isOpened():
        print("Cannot capture from camera. Exiting.")
        quit()
    
    prev_frame_time = 0
    new_frame_time = 0

    while True:
        ret, frame = vs.read()
        if not ret: break

        new_frame_time = time.time()
        show_fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time
        show_fps = int(show_fps)
        show_fps = str("{} FPS".format(show_fps))

        ori_frame = frame

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame, ratio, (dw, dh) = letterbox(frame)

        
