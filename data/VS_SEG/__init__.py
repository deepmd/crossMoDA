from math import sqrt

from .transforms import \
    encoder_train_transforms, encoder_val_transforms, supervised_train_transforms, supervised_val_transforms
from .data import load_data


data_cfg = {
    "input_channels": 1,
    "classes_num": 3,
    "classes": ["Background", "VS", "Cochlea"],
    "max_euclidean_distance_in_volumes": sqrt(512**2 + 512**2 + 40**2)
}


def get_encoder_train_transforms(opt):
    return encoder_train_transforms(opt, "source"), encoder_train_transforms(opt, "target")


def get_encoder_val_transforms(opt):
    return encoder_val_transforms(opt, "source"), encoder_val_transforms(opt, "target")


def get_supervised_train_transforms(opt):
    return supervised_train_transforms(opt, "source"), supervised_train_transforms(opt, "target")


def get_supervised_val_transforms(opt):
    return supervised_val_transforms(opt, "source"), supervised_val_transforms(opt, "target")


def get_data(opt):
    return load_data(opt)
