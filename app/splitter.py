import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from facenet_pytorch import MTCNN
import pyrootutils



pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from app.annotator import Annotator
from app.detector import FaceDetector
from src.models.components.simple_resnet import SimpleResnet


class VideoParser:
    def __init__(self,
                 data_dir,
                 stride=2,
                 detector: FaceDetector = FaceDetector(1, MTCNN()),
                 annotator: Annotator = Annotator(SimpleResnet("resnet18", "DEFAULT", [68,2]), "F:/project/facial_landmarks-wandb/logs/train/runs/2023-04-06_02-25-27\checkpoints\epoch_091.ckpt", None, [224, 224])
                 ) -> None:
        self.parser: cv2.VideoCapture = cv2.VideoCapture()
        self.data_dir = data_dir
        self.video_name = ""
        self.stride = stride
        self.detector = detector
        self.annotator = annotator
        self.index = 0
        self.landmarks = None
        pass

    def feed(self, filename):
        self.video_name = filename[0: len(filename) - 4]
        file_path = os.path.join(self.data_dir, filename)
        self.parser.open(file_path)
        if not self.parser.isOpened():
            print("Cannot open video")
            exit(1)
        return {"name": self.video_name}
    
    def camera(self, index):
        self.parser = cv2.VideoCapture(index)
        self.parser.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.parser.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        # self.parser.set(cv2.CAP_PROP_FPS, 12)
        boxes = []
        while self.parser.isOpened() :
            retval, frame = self.parser.read()
            if retval == False:
                continue
            self.index += 1
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if self.index % self.stride == 0:
                detected, boxes, faces = self.detector.face_detect(image)
                print(boxes)
                if not detected:
                    self.landmarks = None
                    print("Didnt recognize")
                else:
                    bounding_box = [[max(0,int(box[0])), max(0, int(box[1]))] for box in boxes]
                    ratio = [[int(box[2] - bound[0]), 
                              int(box[3] - bound[1])]
                                for box, bound in zip(boxes, bounding_box)]
                    self.landmarks = self.annotator.annotate(faces, bounding_box, ratio)
                    print(self.landmarks)

            if self.landmarks is not None:
                # print(self.landmarks)
                # frame = FaceDetector.show_bounding_box(frame, boxes)
                frame = Annotator.show_landmarks(frame, self.landmarks)
            
            cv2.imshow('frame',frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        self.parser.release()
        cv2.destroyAllWindows()

    
    def parse_video(self):
        output_dir = os.path.join(os.path.join(self.data_dir, "output"), self.video_name)
        os.makedirs(output_dir)
        index = 0
            
if __name__ == "__main__":
    import os
    import hydra
    from omegaconf import OmegaConf, DictConfig
    
    config_path = os.path.join(os.environ["PROJECT_ROOT"], "configs/app")
    @hydra.main(version_base="1.3", config_path=config_path, config_name="splitter.yaml")
    def main(cfg: DictConfig):
        print(OmegaConf.to_yaml(cfg))
        capturer: VideoParser = hydra.utils.instantiate(cfg)
        capturer.camera(0)
    main()