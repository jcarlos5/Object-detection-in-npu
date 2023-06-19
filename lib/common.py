import json

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
