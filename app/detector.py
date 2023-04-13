from types import NoneType
import PIL
from cv2 import INTER_CUBIC
from facenet_pytorch import MTCNN
import cv2
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
import numpy as np
import pyrootutils
import torch
from tqdm.notebook import tqdm



pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from app.annotator import Annotator

class FaceDetector:
    def __init__(self, 
                 resize=1, 
                 mtcnn: MTCNN = MTCNN(),
                 ):
        self.resize = resize
        self.mtcnn = mtcnn
        
    def face_detect(self,image):
        """Detect faces in frames using strided MTCNN."""
        if self.resize != 1:
            image = cv2.resize(image,(int(image.shape[1] * self.resize), int(image.shape[0] * self.resize)), interpolation= cv2.INTER_CUBIC)
        detected = False
        image = cv2.copyMakeBorder(image, 
                                   top=20, 
                                   bottom=20,
                                   left=20,
                                   right=20,
                                   borderType=cv2.BORDER_CONSTANT,
                                   value=[255, 255, 255])
        boxes, probs = self.mtcnn.detect(image)
        if boxes is None:
            detected = False
            return detected, [], []
        detected = True
        faces = []
        # print(len(boxes.shape))
        # print(boxes.shape)
    
        for box in boxes:
            # print(box)
            box = [ max(0, int(b)) for b in box]
            faces.append(image[box[1]:box[3], 
                               box[0]:box[2]])
            print(len(faces))

        # changed back to original size
        # the image can be shrinked, but the bounding box must be original. - 'cause it's explicit and doesnt affect the augmentation
        boxes = boxes / self.resize - 20
        return detected, boxes, faces
    
    #draw on mat
    @staticmethod
    def show_bounding_box(image, boxes):
        for box in boxes:
            image = cv2.rectangle(image, [int(box[0]), int(box[1])], [int(box[2]), int(box[3])], color=(0, 255, 0), thickness=15)
        
        return image

if __name__ == "__main__":
    import os
    import hydra
    from omegaconf import OmegaConf, DictConfig
    
    config_path = os.path.join(os.environ["PROJECT_ROOT"], "configs/app")
    @hydra.main(version_base="1.3", config_path=config_path, config_name="app.yaml")
    def main(cfg: DictConfig):
        # print(OmegaConf.to_yaml(cfg))
        detector: FaceDetector = hydra.utils.instantiate(cfg.detector)
        annotator: Annotator = hydra.utils.instantiate(cfg.annotator)
        if detector is None:
            print("Failed to fetch")
            exit(1)
        
        image = cv2.imread("app/asset/bound_test.jpg")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print(image.shape)
        detected, boxes, faces = detector.face_detect(image)
        print(boxes)
        print(type(faces))
        cloned = []
        cloned.append(cv2.resize(np.array(faces[0]), (224, 224), interpolation=cv2.INTER_CUBIC))
        # image = detector.show_bounding_box(image, boxes=boxes)
        # print(boxes[4])
        # plt.imshow(np.array(faces[4]))
        # plt.show()
        # bounding_box = [[max(0,int(box[0])), max(0, int(box[1]))] for box in boxes]
        # ratio = [[int(min(image.shape[0],box[2]) - bound[0]), int(min(image.shape[1],box[3]) - bound[1])] for box, bound in zip(boxes, bounding_box)]
        pred = annotator.annotate(cloned, [[0, -20]], [[224, 224]])
        np.savetxt("app/asset/filter/base.txt", pred[0])
        print(pred.shape)
        cloned[0] = annotator.show_landmarks(cloned[0], pred)
        plt.imshow(cloned[0])
        plt.show()
    main()