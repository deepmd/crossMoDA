from .transforms import encoder_train_transforms, encoder_val_transforms
from .data import load_data


def get_encoder_train_transforms(opt):
    return encoder_train_transforms(opt, "source"), encoder_train_transforms(opt, "target")


def get_encoder_val_transforms(opt):
    return encoder_val_transforms(opt, "source"), encoder_val_transforms(opt, "target")


def get_data(opt):
    return load_data(opt)
