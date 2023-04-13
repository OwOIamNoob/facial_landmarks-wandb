import os
import string
from typing import List
import PIL
from PIL import ImageDraw
import cv2
from torch import Tensor
import torch
import torchvision
import numpy as np
import pyrootutils
import hydra
from omegaconf import OmegaConf, DictConfig

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.components import simple_resnet
from src.models.dlib_module import DlibLitModule
from src.data.dlib_datamodule import DlibDataModule, TransformDataset

class Annotator:
    def __init__(self,
                 net: torch.nn.Module,
                 ckpt_path,
                 transform: torchvision.transforms.transforms,
                 dimension: List) -> None:
        self.module = DlibLitModule.load_from_checkpoint(ckpt_path, net=net)
        self.transform = transform 
        self.dimension = torch.Tensor(dimension)
        pass

    @torch.no_grad()
    def annotate(self, x, bounding_box, ratio):
        input = []
        # assuming that input is 'faces' = list 
        for i in range(len(x)):
            # in case x is tensor :))
            print("Image size:" + str(x[i].shape))
            transformed = self.transform(image=np.array(x[i]))
            input.append(transformed["image"])
        
        # convert to tensor
        input = torch.stack(input)
        print(input.size())
        # pred is normalize between [-0.5, 0.5] in the [224, 224] window
        pred = self.module.forward(input).detach().numpy()
        for i in range(pred.shape[0]):
            pred[i] = (pred[i] + 0.5) * ratio[i] + bounding_box[i] 
        return pred

    @staticmethod 
    def annotate_image(self,
                       image: PIL.Image, 
                       landmarks: Tensor):
        draw = ImageDraw.Draw(image)
        landmarks = Tensor.numpy(landmarks)
        for i in range(landmarks.shape[0]):
            draw.ellipse((landmarks[i, 0] - 2, landmarks[i, 1] - 2,
                          landmarks[i, 0] + 2, landmarks[i, 1] + 2), fill=(255, 0, 0))
        return image
    
    @staticmethod
    def show_landmarks(image,
                       landmarks: list):
        for i in range(len(landmarks)):
            for index in range(len(landmarks[i])):
                cv2.circle(image, (int(landmarks[i][index][0]), int(landmarks[i][index][1])), radius=3, color=(255, 0, 0), thickness=-1)
        
        return image

if __name__ == "__main__":

    config_path = os.path.join(os.environ["PROJECT_ROOT"], "configs")
    @hydra.main(version_base="1.3", config_path=config_path, config_name="train.yaml")
    def main(cfg: DictConfig):
        annotator: Annotator = hydra.utils.instantiate(cfg.app.annotator)
        datamodule: DlibDataModule = hydra.utils.instantiate(cfg.data)
        datamodule.setup()
        test_dataloader = datamodule.val_dataloader()
        data = next(iter(test_dataloader))
        x, y = data
        pred = annotator.forward_tensor(x, torch.Tensor([0, 0]), torch.Tensor([1, 1]))
        images = TransformDataset.annotate(x, pred)
        torchvision.utils.save_image(images, "C:/Lab/project/facial_landmarks-wandb/output/testing_datamodule_result.png")
        print(images.shape)

    main()
        
