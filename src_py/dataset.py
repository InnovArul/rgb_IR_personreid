import os, sys
import torch
import numpy as np
from skimage import io, transform, img_as_float
import log
import torch.utils.data as data
from utils import *


def read_ids(root_path, split_type):
    config_file_path = os.path.join(root_path, "exp", split_type + "_id.txt")
    with open(config_file_path, "r") as file:
        file_lines = file.readlines()

    # the file has only one line with ids
    id_line = file_lines[0]
    all_ids = ["%04d" % int(i) for i in id_line.split(",")]
    print(config_file_path + " : " + str(len(all_ids)))
    return all_ids


def read_image_of_category(img_path, category):
    # read the image file
    # print(img_path)
    img = img_as_float(io.imread(img_path))

    # if the image is IR, get the single channel as all channels will have same value
    if category == "IR":
        # print('IR first channel')
        img = img[:, :, 1]
        img = img[..., np.newaxis]

    # print('resizing')
    img = transform.resize(img, output_shape=(224, 224))

    # if image is rgb, convert to gray scale
    if category == "rgb":
        # print('gray scale')
        img = rgb2gray(img)
        img = img[..., np.newaxis]

    return img


class cam_ID_folder:
    def __init__(self, root_path, cam_name, ID, cam_config, is_read_image=False):
        # init the instance variables
        self.root_path = root_path
        self.cam_name = cam_name
        self.ID = ID
        self.folder_path = os.path.join(self.root_path, self.cam_name, self.ID)
        self.cam_config = cam_config
        self.is_read_image = is_read_image

    def is_exists(self):
        # returns true if the folder exists
        return os.path.exists(self.folder_path)

    def read_image_file(self, img_path):
        img = None
        if self.is_read_image:
            img = read_image_of_category(self.img_path, self.cam_config[self.cam_name])
        else:
            img = img_path

        return img

    def get_file_instances(self):
        instances = []

        # if the folder exists
        if self.is_exists():
            print(
                self.folder_path
                + " : "
                + str(len(os.listdir(self.folder_path)))
                + " files"
            )
            # for each of the file in the directory
            for file in os.listdir(self.folder_path):
                # read the file and store it in the list
                filepath = os.path.join(self.folder_path, file)
                img = self.read_image_file(filepath)

                instances.append(
                    (img, self.ID, self.cam_config[self.cam_name], self.cam_name)
                )

        return instances


# convert rgb to gray image
def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


class Dataset(data.Dataset):
    def __init__(
        self, root_path, IDs, config_name, is_read_image=True, is_return_path=False
    ):
        self.root_path = root_path
        self.cam_config = {
            "cam1": "rgb",
            "cam2": "rgb",
            "cam3": "IR",
            "cam4": "rgb",
            "cam5": "rgb",
            "cam6": "IR",
        }

        self.IDs = IDs
        self.is_read_image = is_read_image
        self.config_name = config_name
        self.data_instances = self.read_data_instances()
        self.IDs2Classes = {}

        for index, id in enumerate(self.IDs):
            self.IDs2Classes[id] = index

    def read_data_instances(self):
        data_instances = []

        # check if the config already exists
        config_file = os.path.join(self.root_path, self.config_name + "_config.pth")

        # check if the config file is already existing
        if os.path.exists(config_file):
            # load the existing config file
            print("existing config file " + config_file + " found!. Reading the file!")
            data_instances = torch.load(config_file)
        else:
            # for each of the ids
            for ID in self.IDs:
                # for each of the cameras
                for cam_name in self.cam_config.keys():
                    # get all the data instances
                    folder = cam_ID_folder(
                        self.root_path, cam_name, ID, self.cam_config
                    )
                    data_instances += folder.get_file_instances()

            # save the configuration
            torch.save(data_instances, config_file)

        return data_instances

    def __len__(self):
        return len(self.data_instances)

    def do_random_translation(self, img):
        height, width, _ = img.shape
        width_range = 0.1 * width
        height_range = 0.1 * height

        # get a random height and width
        translate_height = np.random.uniform(-height_range, height_range)
        translate_width = np.random.uniform(-width_range, width_range)

        # create a similarity transform
        sim_transform = transform.SimilarityTransform(
            scale=1, rotation=0, translation=(translate_width, translate_height)
        )

        # warp the image
        img = transform.warp(img, sim_transform)
        return img

    def pad_zeros_by_category(self, img, category):
        # pad the zeros based on img type
        # print(img.shape)
        padding = np.zeros_like(img)
        if category == "rgb":
            # for rgb images, pad zeros as second channel
            img = np.concatenate((img, padding), axis=2)
        else:
            # for IR images, pad zeros as first channel
            img = np.concatenate((padding, img), axis=2)
        return img

    def __getitem__(self, index):
        img, ID, category, cam_name = self.data_instances[index]

        if self.is_read_image:
            img = read_image_of_category(img, category)

        # do random data augmentation
        img = self.do_random_translation(img)

        # pad zeros according to category
        img = self.pad_zeros_by_category(img, category)

        return (
            img.transpose((2, 0, 1)),
            self.IDs2Classes[ID],
        )  # for data.Dataset .transpose((2,0,1))


class TestDataset(Dataset):
    def __init__(self, root_path, IDs, config_name):
        # init the super class
        super().__init__(root_path, IDs, config_name)

    def read_data_instances(self):
        print("child class method")
        data_instances = {}
        for cam_name in self.cam_config.keys():
            data_instances[cam_name] = {}

        # check if the config already exists
        config_file = os.path.join(self.root_path, self.config_name + "_config.pth")

        # check if the config file is already existing
        if os.path.exists(config_file):
            # load the existing config file
            print("existing config file " + config_file + " found!. Reading the file!")
            data_instances = torch.load(config_file)
        else:
            # for each of the ids
            for ID in self.IDs:
                # for each of the cameras
                for cam_name in self.cam_config.keys():
                    # get all the data instances
                    folder = cam_ID_folder(
                        self.root_path, cam_name, ID, self.cam_config
                    )
                    current_folder_instances = folder.get_file_instances()
                    current_folder_instances = self.order_file_names(
                        current_folder_instances
                    )
                    data_instances[cam_name][ID] = current_folder_instances

            # save the configuration
            torch.save(data_instances, config_file)

        return data_instances

    def get_cam_files_config(self):
        return self.data_instances

    def read_image_from_config(self, config):
        # config will contain
        # img path, ID, category(rgb or IR), cam name
        img = read_image_of_category(config[0], config[2])
        img = self.pad_zeros_by_category(img, config[2])
 
        return torch.from_numpy(img.transpose((2, 0, 1)))

    def order_file_names(self, instances):
        # create a hash with file name
        filenames_hash = {}
        for inst in instances:
            filename = get_file_name(inst[0])
            # print(filename)
            filenames_hash[filename] = inst

        # create an array ordered by filename, in numerical order
        total_files = len(instances)
        ordered_instances = []
        for i in range(total_files):
            ordered_instances.append(filenames_hash["%04d" % (i + 1)])

        return ordered_instances
