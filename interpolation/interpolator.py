import torch
from torch.nn import functional as F
import warnings
warnings.filterwarnings("ignore")

import logging


from dataclasses import dataclass

@dataclass
class InterpolatorConfig:
    use_interpolation: bool
    model_name: str
    model_weights_path: str
    exp: int
    padding_divider: int


class Interpolator():

    def __init__(self, cfg: InterpolatorConfig):

        torch.set_grad_enabled(False)
        if torch.cuda.is_available():
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True


        self.use_interpolation = cfg.use_interpolation
        self.model_name = cfg.model_name
        self.model_weights_path = cfg.model_weights_path
        self.exp = cfg.exp
        self.padding_divider = cfg.padding_divider

        self.model = None

        if self.use_interpolation:

            logging.info(f"interpolation is used with {self.exp}")

            match self.model_name:
                case "RIFEv4.25lite_1018":
                    from interpolation.rife_model.RIFE_HDv3_practical import Model
                    self.model = Model()

                case "RIFE_trained_v6":
                    from interpolation.rife_model.RIFE import Model
                    self.model = Model()

                case "RIFE_trained_model_v3.6":
                    from interpolation.rife_model.RIFE_HDv3 import Model
                    self.model = Model()


                case _:
                   logging.error("Wrong model name")

            if self.model is not None:
                try:
                    self.model.load_model(self.model_weights_path, -1)
                    logging.info(f"Loaded interpolation model {self.model_name}")
                    self.model.eval()
                    self.model.device()
                    
                except Exception as exception:
                    self.model = None
                    logging.error(f"Cant load model {exception=}")

        if not self.use_interpolation or (self.model is None):
            logging.info(f"interpolation is not used")


    def interpolate_frames(self, first_frame, second_frame):

        img0 = first_frame
        img1 = second_frame

        n, c, h, w = img0.shape
        ph = ((h - 1) // self.padding_divider + 1) * self.padding_divider
        pw = ((w - 1) // self.padding_divider + 1) * self.padding_divider
        padding = (0, pw - w, 0, ph - h)
        img0 = F.pad(img0, padding)
        img1 = F.pad(img1, padding)

        img_list = [img0, img1]

        if self.use_interpolation and (self.model is not None):

            for i in range(self.exp):
                tmp = []
                for j in range(len(img_list) - 1):
                    mid = self.model.inference(img_list[j], img_list[j + 1])
                    tmp.append(img_list[j])
                    tmp.append(mid)
                tmp.append(img1)
                img_list = tmp

        return [x[..., :h, :w] for x in img_list]