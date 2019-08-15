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
 

force_new_model = True
pretrained_model = None
# init the settings
settings.init_settings(force_new_model, pretrained_model)
# init the log
log.init_logger(tensorboard=False)


def show_images(img):
    show_image(img[:, :, 0])
    show_image(img[:, :, 1])


def train(model, train_data, criterion, optimizer, epoch):
    total_train_images = len(train_data)
    # print(total_train_images)

    if opt["useGPU"]:
        model = model.cuda()

    # training mode
    model.train()

    logger.info("epoch # " + str(epoch))
    total_loss = 0

    # for index in range(total_train_images):
    #     img, classID = train_data[100+index]
    #     show_images(img)
    #     input()

    for batch_index, contents in enumerate(tqdm(train_data)):
        imgs, targets = contents

        if opt["useGPU"]:
            imgs = imgs.float().cuda()
            targets = targets.cuda()

        var_imgs = Variable(imgs)
        var_targets = Variable(targets)

        features, output = model(var_imgs)

        optimizer.zero_grad()
        loss = criterion(output, var_targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if batch_index % 100 == 0:
            logger.info("total loss for current epoch so far : " + str(total_loss))

    logger.info("total loss for current epoch : " + str(total_loss))
    torch.save(
        model.cpu().state_dict(),
        os.path.join(opt["save"], "deep_zero_model#" + str(epoch) + ".pth"),
    )


if __name__ == "__main__":
    opt = settings.opt
    logger = log.logger
    train_ids = read_ids(opt["dataroot"], "train")
    val_ids = read_ids(opt["dataroot"], "val")
    test_ids = read_ids(opt["dataroot"], "test")

    train_dataset = Dataset(opt["dataroot"], train_ids, "train")
    val_dataset = Dataset(opt["dataroot"], val_ids, "val")
    test_dataset = TestDataset(opt["dataroot"], test_ids, "test")

    train_dataloader = data.DataLoader(
        train_dataset, batch_size=opt["batch_size"], shuffle=True
    )

    model = get_model(
        arch=opt["arch"], num_classes=len(train_ids), pretrained_model=pretrained_model
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=opt["lr"],
        momentum=opt["momentum"],
        nesterov=opt["nesterov"],
        weight_decay=opt["weight_decay"],
    )

    # for each epoch, run the training
    for epoch in range(opt["epochs"]):
        train(model, train_dataloader, criterion, optimizer, epoch)
