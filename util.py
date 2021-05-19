import datetime
import logging
import math
import os

import numpy as np
import torch
import torch.optim as optim


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


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


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


def set_optimizer(opt, model):
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    return optimizer


def save_model(model, optimizer, opt, epoch, save_file):
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
    formatter = logging.Formatter("%(asctime)s,%(msecs)03d %(levelname)s        %(message)s", datefmt="%H:%M:%S")
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
