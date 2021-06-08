import datetime
import logging
import math
import os
from operator import itemgetter, mod

import numpy as np
import torch


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def load_model(model, opt, load_file):
    opt.logger.info(f'==> Loading... "{load_file}"')
    ckpt = torch.load(opt.ckpt, map_location='cpu')
    state_dict = ckpt['model']
    not_found_keys = [key for key in model.state_dict() if key not in state_dict]
    model.load_state_dict(state_dict, strict=False)
    if len(not_found_keys) > 0:
        opt.logger.warning(f"Missing key(s) in checkpoint file: {not_found_keys}")


def save_checkpoint(model, optimizer, opt, epoch, save_file):
    opt.logger.info(f'==> Saving... "{save_file}"')
    opt_dict = {k: v for k, v in opt.__dict__.items() if k != "logger"}
    state = {
        'opt': opt_dict,
        'model': strip_DataParallel(model).state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state


def strip_DataParallel(net):
    if isinstance(net, torch.nn.DataParallel):
        return strip_DataParallel(net.module)
    return net


def set_up_logger(logs_path, log_file_name=None):
    # logging settings
    logger = logging.getLogger()
    if log_file_name is None:
        log_file_name = str(datetime.datetime.now()).split(".")[0] \
            .replace(" ", "_").replace(":", "_").replace("-", "_") + ".log"
    fileHandler = logging.FileHandler(os.path.join(logs_path, log_file_name), mode="w")
    consoleHandler = logging.StreamHandler()
    logger.addHandler(fileHandler)
    logger.addHandler(consoleHandler)
    formatter = logging.Formatter("%(asctime)s,%(msecs)03d %(levelname).1s   %(message)s", datefmt="%H:%M:%S")
    fileHandler.setFormatter(formatter)
    consoleHandler.setFormatter(formatter)
    logger.setLevel(logging.INFO)
    logger.info("Created " + log_file_name)
    return logger


def log_parameters(opt, logger):
    logger.info("-" * 10)
    logger.info("Parameters: ")
    opt_dict = opt.__dict__
    longest_key = max(len(k) for k in opt_dict.keys())
    for name, value in opt_dict.items():
        logger.info(f"{name.ljust(longest_key)} = {value}")
    logger.info("-" * 10)
