# import matplotlib.pyplot as plt
import os, sys, torch
from model import resnet6
#from model_resnet_pretrained import ResNet50
from model_resnet_bam import ResNet50


def show_image(image):
    dpi = 80
    figsize = (image.shape[1] / float(dpi), image.shape[0] / float(dpi))
    fig = plt.figure(figsize=figsize)
    plt.imshow(image)
    fig.show()


def get_file_name(filepath):
    return os.path.basename(filepath).split(".")[0]


def get_model(arch, num_classes, pretrained_model):
    if arch == "resnet6":
        model = resnet6(num_classes=num_classes)
    elif arch == "resnet50":
        model = ResNet50(num_classes=num_classes)
    else:
        assert False, "unknown model arch: " + arch

    if pretrained_model is not None:
        model.load_state_dict(torch.load(pretrained_model))

    return model


def load_pretrained_model(model, pretrained_model_path, verbose=False):
    """To load the pretrained model considering the number of keys and their sizes
    
    Arguments:
        model {loaded model} -- already loaded model
        pretrained_model_path {str} -- path to the pretrained model file
    
    Raises:
        IOError -- if the file path is not found
    
    Returns:
        model -- model with loaded params
    """

    if isinstance(pretrained_model_path, str):
        if not os.path.exists(pretrained_model_path):
            raise IOError(
                "Can't find pretrained model: {}".format(pretrained_model_path)
            )

        print("Loading checkpoint from '{}'".format(pretrained_model_path))
        pretrained_state = torch.load(pretrained_model_path)["state_dict"]
    else:
        # incase pretrained model weights are given
        pretrained_state = pretrained_model_path

    print(len(pretrained_state), " keys in pretrained model")

    current_model_state = model.state_dict()
    print(len(current_model_state), " keys in current model")
    pretrained_state = {
        key: val
        for key, val in pretrained_state.items()
        if key in current_model_state and val.size() == current_model_state[key].size()
    }

    print(
        len(pretrained_state),
        " keys in pretrained model are available in current model",
    )
    current_model_state.update(pretrained_state)
    model.load_state_dict(current_model_state)

    if verbose:
        non_available_keys_in_pretrained = [
            key
            for key, val in pretrained_state.items()
            if key not in current_model_state
            or val.size() != current_model_state[key].size()
        ]
        non_available_keys_in_current = [
            key
            for key, val in current_model_state.items()
            if key not in pretrained_state or val.size() != pretrained_state[key].size()
        ]

        print(
            "not available keys in pretrained model: ", non_available_keys_in_pretrained
        )
        print("not available keys in current model: ", non_available_keys_in_current)

    return model
