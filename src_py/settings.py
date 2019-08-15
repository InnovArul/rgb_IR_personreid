# options file
import time, os, sys, torch, torchvision
import torchnet.meter as meter
from pathlib import Path
from torch.autograd import Variable
import torch


def init_settings(force_new_model, pretrained_model_path=None):
    global opt
    opt = dict()
    opt["dataroot"] = "../datasets/sysu_mm01"
    opt["batch_size"] = 64
    opt["lr"] = 0.001
    opt["momentum"] = 0.9
    opt["weight_decay"] = 0.0005
    opt["nesterov"] = False
    opt["description"] = "deep-zero-padding"
    opt["useGPU"] = True
    opt["epochs"] = 20000
    opt["arch"] = "resnet50"  # resnet6 | resnet50

    # determine the log / save folder path
    if force_new_model:
        opt["save"] = (
            "../scratch/sysu_mm01/deepzeropadding-"
            + time.strftime("%d%b%Y-%H%M%S")
            + "_"
            + opt["description"]
        )
    else:
        opt["save"] = str(Path(pretrained_model_path).parent)
    print("save path : " + opt["save"])
    os.makedirs(opt["save"], exist_ok=True)

