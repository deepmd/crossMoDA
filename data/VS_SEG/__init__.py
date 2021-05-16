from .transforms import get_encoder_source_transforms, get_encoder_target_transforms
from .data import load_data


def get_encoder_transforms(opt):
    return get_encoder_source_transforms(opt), get_encoder_target_transforms(opt)


def get_data(opt):
    return load_data(opt)
