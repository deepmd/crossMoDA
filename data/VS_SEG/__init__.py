from .transforms import encoder_transforms
from .data import load_data


def get_encoder_transforms(opt):
    return encoder_transforms(opt, "source"), encoder_transforms(opt, "target")


def get_data(opt):
    return load_data(opt)
