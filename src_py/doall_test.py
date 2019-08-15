import os, sys, torch
import numpy as np
from torch.autograd import Variable
from dataset import *
import settings, log
import torch.nn as nn, torch.utils.data as data
import torch.optim as optim
from utils import *
from tqdm import tqdm
from model import resnet6
import scipy.io as sio

pretrained_model = "../scratch/sysu_mm01/deepzeropadding-14May2019-125214_deep-zero-padding/deep_zero_model#156.pth"
# init the settings

settings.init_settings(False, pretrained_model)
# init the log
log.init_logger(tensorboard=False, prepend_text="test_")


def get_max_test_id(test_ids):
    int_test_ids = [int(ID) for ID in test_ids]
    return np.max(int_test_ids)


def prepare_empty_matfile_config(max_test_id):
    cam_features = np.empty(max_test_id, dtype=object)
    for i in range(len(cam_features)):
        cam_features[i] = []
    return cam_features


def test(model, test_dataset, test_ids):
    data_instances = test_dataset.get_cam_files_config()
    #print(len(data_instances), data_instances[0])
    matfile_prefix = get_file_name(pretrained_model)
    testresults_dir = os.path.join(opt["save"], matfile_prefix)
    if not os.path.exists(testresults_dir):
        os.mkdir(testresults_dir)

    max_test_id = get_max_test_id(test_ids)
    model.eval()
    if opt["useGPU"]:
        model = model.cuda()

    for cam_name, id_contents in data_instances.items():
        # data_instances
        #  --cam1
        #      -- 0001
        #           -- 0001.jpg
        #           -- 0002.jpg
        #           -- ...
        matfile_path = os.path.join(
            testresults_dir, matfile_prefix + "_" + cam_name + ".mat"
        )
        print(cam_name)
        
        # prepare empty features for all the person ids upto max_test_id
        # in the feature_original, features wrt all the images from all ids are extracted regardless 
        # of whether it is a test subject or not
        # but this script only extracts the features for test subjects, that too only upto max_test_id (cam3 contains 533 ids, but 333 is the max test id)
        # other ids within 333 will have empty features (shape = (0,0))
        cam_features = prepare_empty_matfile_config(max_test_id)

        for id_, img_contents in tqdm(id_contents.items()):
            all_current_id_features = np.empty(shape=[0, 2048])
            for img_config in img_contents:
                #print(img_config)
                img = test_dataset.read_image_from_config(img_config)

                if opt["useGPU"]:
                    img = img.unsqueeze(0).float().cuda()

                var_img = Variable(img)
                features, _ = model(var_img)
                current_feature = features.data[0].cpu().numpy().reshape(1, -1)
                #print(current_feature.shape)
                #print(all_current_id_features.shape)
                all_current_id_features = np.append(
                    all_current_id_features, current_feature, axis=0
                )

            cam_features[int(id_) - 1] = all_current_id_features
            # print(cam_features[int(id_)-1].shape)

        sio.savemat(matfile_path, {"feature": cam_features})


if __name__ == "__main__":
    opt = settings.opt
    logger = log.logger
    train_ids = read_ids(opt["dataroot"], "train")
    test_ids = read_ids(opt["dataroot"], "test")
    test_dataset = TestDataset(opt["dataroot"], test_ids, "test")

    model = get_model(num_classes=len(train_ids), arch = opt['arch'], pretrained_model = pretrained_model)
    model.load_state_dict(torch.load(pretrained_model))

    test(model, test_dataset, test_ids)
