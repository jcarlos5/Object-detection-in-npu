from lib.common import SETTINGS, open_cam, draw
from lib.process import nms, box_convert
from rknnlite.api import RKNNLite
from imutils.video import FPS
import numpy as np
import time
import cv2
        

if __name__ == '__main__':
    SETTINGS.load("config.json")

    rknn_lite = RKNNLite()
    #ret = rknn_lite.load_rknn(SETTINGS.MODEL_PATH)
    ret = rknn_lite.load_onnx(SETTINGS.MODEL_ONNX_PATH)
    if ret != 0: exit(ret)
    ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0)

    vs = open_cam()
    fps = FPS().start()

    if not vs.isOpened():
        print("Cannot capture from camera. Exiting.")
        quit()
    
    prev_ft = 0
    new_ft = 0

    while True:
        ret, frame = vs.read()
        if not ret: break

        new_ft = time.time()
        show_fps = f"{int(1/(new_ft-prev_ft))} FPS"
        prev_ft = new_ft
        
        fh, fw = frame.shape[:2]
        frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), tuple(SETTINGS.IMAGE_SIZE))
        outputs = rknn_lite.inference(inputs=[np.expand_dims(frame,axis=0)], data_format='nhwc')[0][0]
        outputs = outputs.reshape(outputs.shape[:2])
        
        filter_output = nms(outputs, 0.8, 0.5)
        print(filter_output)
        if filter_output.shape[0] != 0:
            bboxes = box_convert(filter_output[:, 0:4])
            scores = filter_output[:, 4].tolist()
            classes = filter_output[:, 5].tolist()
            draw(frame, bboxes, scores, classes)

        frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),(fw, fh))
        cv2.putText(frame, show_fps, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 1, cv2.LINE_AA)
        cv2.imshow("OBJETOS DETECTADOS", frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord("q"): break
        fps.update()
    
    #rknn_lite.release()
    rknn_lite.release()

    # stop the timer and display FPS information
    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    
    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.release()
        