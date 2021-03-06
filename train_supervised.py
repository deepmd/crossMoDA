import os
import argparse
import time
import math
from collections import defaultdict
from functools import partial
from operator import itemgetter

import numpy as np
import monai
import torch
import torch.backends.cudnn as cudnn
from monai.losses import DiceLoss
from torch.utils.data import ConcatDataset, WeightedRandomSampler

from losses import HardnessWeightedDiceLoss
from torch.utils.tensorboard import SummaryWriter
from monai.data import DataLoader
from matplotlib import pyplot as plt
from tqdm import tqdm

from metrics import compute_dice_score, compute_average_symmetric_surface_distance
from optimizers import get_optimizer
from util import \
    AverageMeter, adjust_learning_rate, warmup_learning_rate, save_checkpoint, load_model, set_up_logger, log_parameters
from networks import SegAuxModel
from data.datasets import CacheSliceDataset
from data.VS_SEG import get_supervised_train_transforms, get_supervised_val_transforms, get_data, data_cfg


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument("--debug", dest="debug", action="store_true",
                        help="activate debugging mode")
    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=500,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--optimizer', type=str, default='SGD',
                        choices=['SGD', 'Adam'], help='optimizer')
    parser.add_argument('--learning_rate', type=float, default=0.05,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='350,400,450',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str,
                        choices=['SegResNet', 'ResUNet', 'ResUNet++'])
    parser.add_argument('--dataset', type=str, default='VS_SEG',
                        choices=['VS_SEG', 'crossMoDA'], help='dataset')
    parser.add_argument("--split", type=str, default="split.csv",
                        help="path to CSV file that defines training, validation and test datasets")
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    parser.add_argument('--size', type=int, default=384, help='expected size of input')
    parser.add_argument('--include_target', action='store_true',
                        help='include target domain images and labels in training')
    parser.add_argument('--fg_weighted', action='store_true',
                        help='sample training slices based on their foreground ratio')
    parser.add_argument('--fg_thresh', type=float, default=None,
                        help='foreground ratio threshold to filter training slices')

    # other setting
    parser.add_argument('--loss', type=str, default="HDLoss",
                        choices=['HDLoss', 'DiceLoss'])
    parser.add_argument('--hardness_lambda', type=float, default=0.6,
                        help='controls the degree of hard voxel weighting in Hardness-Weighted Dice Loss (HDLoss)')
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')

    # pretrain model
    parser.add_argument('--ckpt', type=str, default=None,
                        help='path to pre-trained model')
    parser.add_argument('--freeze_enc', action='store_true',
                        help='freeze encoder')

    opt = parser.parse_args()

    if opt.data_folder is None:
        opt.data_folder = './datasets/'

    opt.in_channels = data_cfg["input_channels"]
    opt.classes_num = data_cfg["classes_num"]
    opt.classes = data_cfg["classes"]

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = f"{opt.dataset}_sup_{opt.model}_{opt.loss}_{opt.optimizer}_lr_{opt.learning_rate}_" + \
                     f"decay_{opt.weight_decay}_bsz_{opt.batch_size}"

    if opt.fg_thresh is not None:
        opt.model_name += f"_fgtr_{opt.fg_thresh}"
    if opt.fg_weighted:
        opt.model_name += "_fgw"

    if opt.cosine:
        opt.model_name += "_cosine"

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name += "_warm"
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    if opt.include_target:
        opt.model_name += "_target"

    opt.model_name += f"_trial_{opt.trial}"

    save_path = os.path.join("./save", opt.model_name)

    opt.tb_folder = os.path.join(save_path, "tensorboard")
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(save_path, "models")
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    opt.log_folder = os.path.join(save_path, "logs")
    if not os.path.isdir(opt.log_folder):
        os.makedirs(opt.log_folder)

    return opt


def _get_foreground_ratio(data):
    label = data["label"]
    total_area = label.shape[0] * label.shape[1]
    if isinstance(label, np.ndarray):
        label = np.sum(label, axis=1)
        foreground = (label > 0).astype(float)
        foreground = np.sum(foreground, axis=(0, 1)) / total_area
        return foreground
    elif isinstance(label, torch.Tensor):
        label = torch.sum(label, dim=0)
        foreground = (label > 0).float()
        foreground = torch.sum(foreground, dim=(0, 1)) / total_area
        return foreground
    else:
        raise TypeError(f"Cannot calculate foreground ratio for type {type(label).__name__}.")


def _filter_slices_by_fg_ratio(data, threshold):
    fg_ratio = _get_foreground_ratio(data)
    return fg_ratio > threshold


def _weight_slices_by_fg_ratio(data):
    fg_ratio = _get_foreground_ratio(data)
    # weights = (fg_ratio + 1e-10) ** 0.1
    weigths = torch.ones(len(fg_ratio))
    weigths[fg_ratio > 0] *= 7
    return weigths


def set_loader(opt):
    logger = opt.logger
    source_train_transform, target_train_transform = get_supervised_train_transforms(opt)
    source_val_transform, target_val_transform = get_supervised_val_transforms(opt)
    (source_train_data, target_train_data), (source_val_data, target_val_data), _ = get_data(opt)

    # train data
    filter_slices = None
    if opt.fg_thresh is not None:
        filter_slices = partial(_filter_slices_by_fg_ratio, threshold=opt.fg_thresh)
    weight_slices = None
    if opt.fg_weighted:
        weight_slices = _weight_slices_by_fg_ratio

    logger.info("Caching source training data ...")
    train_dataset = CacheSliceDataset(data=source_train_data, transform=source_train_transform,
                                      keys=["image", "label"], num_workers=opt.num_workers//3,
                                      filter_slices=filter_slices, weight_slices=weight_slices)
    weights = train_dataset.get_weights()
    if opt.include_target:
        logger.info("Caching target training data ...")
        target_train_dataset = CacheSliceDataset(data=target_train_data, transform=target_train_transform,
                                                 keys=["image", "label"], num_workers=opt.num_workers//3,
                                                 filter_slices=filter_slices, weight_slices=weight_slices)
        weights = torch.cat([weights, target_train_dataset.get_weights()])
        train_dataset = ConcatDataset([train_dataset, target_train_dataset])

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=opt.batch_size,
        sampler=WeightedRandomSampler(weights, num_samples=len(weights)) if opt.fg_weighted else None,
        shuffle=not opt.fg_weighted,
        num_workers=opt.num_workers,
        pin_memory=True)
    logger.info(f"Summary of the training data:")
    logger.info(f"number of slices  = {len(train_dataset)}")
    logger.info(f"number of batches = {len(train_loader)}")

    # validation data
    logger.info("Caching source validation data ...")
    source_val_dataset = CacheSliceDataset(data=source_val_data, transform=source_val_transform,
                                           keys=["image", "label"], num_workers=opt.num_workers//3)
    logger.info("Caching target validation data ...")
    target_val_dataset = CacheSliceDataset(data=target_val_data, transform=target_val_transform,
                                           keys=["image", "label"], num_workers=opt.num_workers//3)
    source_val_loader = DataLoader(
        dataset=source_val_dataset,
        batch_size=opt.batch_size,
        shuffle=False,  # important otherwise cannot create volumes out of slices in validate
        num_workers=opt.num_workers,
        pin_memory=True)
    target_val_loader = DataLoader(
        dataset=target_val_dataset,
        batch_size=opt.batch_size,
        shuffle=False,  # important otherwise cannot create volumes out of slices in validate
        num_workers=opt.num_workers,
        pin_memory=True)
    logger.info(f"Summary of the validation data:")
    logger.info(f"number of slices  (source/target) = {len(source_val_dataset)}/{len(target_val_dataset)}")
    logger.info(f"number of batches (source/target) = {len(source_val_loader)}/{len(target_val_loader)}")

    return train_loader, source_val_loader, target_val_loader


def set_model(opt):
    model = SegAuxModel(
        mode="encoder+decoder",
        base_args={
            "in_channels": opt.in_channels,
            "classes_num": opt.classes_num,
            "model": opt.model,
            "size": opt.size,
        }
    )

    if opt.loss == "HDLoss":
        criterion = HardnessWeightedDiceLoss(to_onehot_y=True, softmax=True, hardness_lambda=opt.hardness_lambda)
    elif opt.loss == "DiceLoss":
        criterion = DiceLoss(to_onehot_y=True, softmax=True)
    else:
        raise ValueError(f"Specified loss name '{opt.loss}' is not valid.")

    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if opt.ckpt is not None:
        load_model(model.base, opt, opt.ckpt)

    if opt.freeze_enc:
        model.base.freeze_encoder()

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    optimizer = get_optimizer(model.parameters(), opt)

    return model, criterion, optimizer


def check_data(data_loader, opt, name, max_samples=None):
    logger = opt.logger
    samples_path = os.path.join(opt.log_folder, f"{name}_samples")
    if not os.path.isdir(samples_path):
        os.makedirs(samples_path)
    logger.info(f"Saving {name} sample data to {samples_path}")
    max_samples = len(data_loader.dataset) if max_samples is None else max_samples
    samples_num = 0
    for idx, batch_data in enumerate(tqdm(data_loader, desc="Saving  samples"), start=1):
        bsz = batch_data["image"].shape[0]
        for i in range(bsz):
            samples_num += 1
            if samples_num > max_samples:
                continue
            image_orig = batch_data["image_orig"][i, 0, ...]
            image = batch_data["image"][i, 0, ...]
            label = batch_data["label"][i, 0, ...]
            vol_idx = batch_data["vol_idx"][i].item()
            slice_idx = batch_data["slice_idx"][i].item()
            file_name = batch_data["image_meta_dict"]["filename_or_obj"][i]
            fig = plt.figure("check", (3*6, 6))
            fig.suptitle(f"Batch:{idx}  Item:{i+1}  Volume:{vol_idx+1} Slice:{slice_idx+1}  "
                         f"File:{file_name}", fontsize=16)
            plt.subplot(1, 3, 1)
            plt.title(f"Original Image {tuple(image_orig.shape)}")
            plt.imshow(image_orig, cmap="gray", interpolation="none")
            plt.subplot(1, 3, 2)
            plt.title(f"Image {tuple(image.shape)}")
            plt.imshow(image, cmap="gray", interpolation="none")
            plt.subplot(1, 3, 3)
            plt.title(f"Labels on image {tuple(label.shape)}")
            plt.imshow(image, cmap="gray", interpolation="none")
            plt.imshow(label, cmap="jet", interpolation="none", vmin=0, vmax=opt.classes_num-1, alpha=0.5)
            plt.savefig(os.path.join(samples_path, f"batch_{idx}_item_{i+1}_vol_{vol_idx+1}_slice_{slice_idx+1}.png"))


def train(train_loader, model, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    dscs = AverageMeter()

    end = time.time()
    for idx, batch_data in enumerate(train_loader):
        data_time.update(time.time() - end)

        bsz = batch_data["image"].shape[0]
        images = batch_data["image"]
        labels = batch_data["label"]

        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        logits = model(images)["decoder_output"]
        loss = criterion(logits, labels)

        # update metric
        with torch.no_grad():
            losses.update(loss.item(), bsz)
            dsc, valid_n = compute_dice_score(logits, labels)
            dscs.update(dsc, valid_n)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            opt.logger.info(f"[{epoch}][{idx+1}/{len(train_loader)}]\t" +
                f"BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t" +
                f"DT {data_time.val:.3f} ({data_time.avg:.3f})\t" +
                f"Loss {losses.val:.3f} ({losses.avg:.3f})\t" +
                f"DSC {dscs.val[1:].mean().item():.3f} <{', '.join(f'{m.item():.3f}' for m in dscs.val)}> " +
                f"({dscs.avg[1:].mean().item():.3f} <{', '.join(f'{m.item():.3f}' for m in dscs.avg)}>)")

    return losses.avg, (dscs.avg[1:].mean().item(), {cls: m for cls, m in zip(opt.classes, dscs.avg)})


def validate(source_val_loader, target_val_loader, model, opt):
    """validation"""
    model.eval()

    source_dscs = AverageMeter()
    source_assds = AverageMeter()
    target_dscs = AverageMeter()
    target_assds = AverageMeter()

    items = ((source_val_loader, source_dscs, source_assds), (target_val_loader, target_dscs, target_assds))

    with torch.no_grad():
        for data_loader, dscs, assds in items:
            volume_logits = defaultdict(lambda: [])
            volume_labels = defaultdict(lambda: [])
            for batch_data in data_loader:
                images = batch_data["image"]
                labels = batch_data["label"]
                vol_idxs = batch_data["vol_idx"].squeeze()
                if torch.cuda.is_available():
                    images = images.cuda(non_blocking=True)
                    labels = labels.cuda(non_blocking=True)
                logits = model(images)
                unique_vol_idxs = torch.unique(vol_idxs)
                for vol_idx in unique_vol_idxs.tolist():
                    volume_logits[vol_idx].append(logits["decoder_output"][vol_idxs == vol_idx])
                    volume_labels[vol_idx].append(labels[vol_idxs == vol_idx])
                if len(volume_logits) > 1:
                    for vol_idx in sorted(volume_logits.keys())[:-1]:
                        vol_logits = torch.cat(volume_logits.pop(vol_idx))
                        vol_logits = torch.movedim(vol_logits, 0, -1)
                        vol_logits = torch.unsqueeze(vol_logits, dim=0)
                        vol_labels = torch.cat(volume_labels.pop(vol_idx))
                        vol_labels = torch.movedim(vol_labels, 0, -1)
                        vol_labels = torch.unsqueeze(vol_labels, dim=0)
                        dsc, valid_n = compute_dice_score(vol_logits, vol_labels)
                        if 0 in valid_n:
                            empty_cls_idxs = (valid_n == 0).nonzero(as_tuple=True)[0].tolist()
                            opt.logger.warn("There are no instance of classes(s) " +
                                            f"{itemgetter(*empty_cls_idxs)(opt.classes)} in volume {vol_idx+1}.")
                        dscs.update(dsc, valid_n)
                        assd, valid_n = compute_average_symmetric_surface_distance(vol_logits, vol_labels)
                        assds.update(assd, valid_n)

        return (source_dscs.avg[1:].mean().item(), {cls: m for cls, m in zip(opt.classes, source_dscs.avg)}), \
               (source_assds.avg[1:].mean().item(), {cls: m for cls, m in zip(opt.classes, source_assds.avg)}), \
               (target_dscs.avg[1:].mean().item(), {cls: m for cls, m in zip(opt.classes, target_dscs.avg)}), \
               (target_assds.avg[1:].mean().item(), {cls: m for cls, m in zip(opt.classes, target_assds.avg)})


def main():
    opt = parse_option()

    # Set deterministic training for reproducibility
    monai.utils.set_determinism(2147483647)
    cudnn.deterministic = False  # due to performance issues

    # logger
    logger = set_up_logger(logs_path=opt.log_folder)
    log_parameters(opt, logger)
    opt.logger = logger

    # tensorboard
    tb_logger = SummaryWriter(log_dir=opt.tb_folder, flush_secs=30)

    # build data loader
    train_loader, source_val_loader, target_val_loader = set_loader(opt)

    # build model, criterion and optimizer
    model, criterion, optimizer = set_model(opt)

    if opt.debug:
        check_data(train_loader, opt, "train", max_samples=200)
        check_data(source_val_loader, opt, "source_val", max_samples=30)
        check_data(target_val_loader, opt, "target_val", max_samples=30)

    max_dist = data_cfg["max_euclidean_distance_in_volumes"]

    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss, dsc = train(train_loader, model, criterion, optimizer, epoch, opt)
        time2 = time.time()
        source_dsc, source_assd, target_dsc, target_assd = validate(source_val_loader, target_val_loader, model, opt)
        time3 = time.time()
        logger.info(f"Epoch {epoch}  Train: Time {(time2 - time1):.2f},  " +
                    f"Loss {loss:.3f},  DSC {dsc[0]:.3f}")
        logger.info(f"{' '*(len(str(epoch))+1)}  Validation: Time {(time3 - time2):.2f},  " +
                    f"S_DSC {source_dsc[0]:.3f} <{', '.join(f'{m:.3f}' for m in source_dsc[1].values())}>,  "
                    f"S_ASSD {source_assd[0]:.1f} <{', '.join(f'{m:.1f}' for m in source_assd[1].values())}>,  "
                    f"T_DSC {target_dsc[0]:.3f} <{', '.join(f'{m:.3f}' for m in target_dsc[1].values())}>,  "
                    f"T_ASSD {target_assd[0]:.1f} <{', '.join(f'{m:.1f}' for m in target_assd[1].values())}>")

        # tensorboard logging
        tb_logger.add_scalar('train/loss', loss, epoch)
        tb_logger.add_scalar('train/mean_dsc', dsc[0], epoch)
        tb_logger.add_scalar('val_source/mean_dsc', source_dsc[0], epoch)
        tb_logger.add_scalar('val_source/mean_assd', min(source_assd[0], max_dist), epoch)
        tb_logger.add_scalar('val_target/mean_dsc', target_dsc[0], epoch)
        tb_logger.add_scalar('val_target/mean_assd', min(target_assd[0], max_dist), epoch)
        for cls in dsc[1].keys():
            tb_logger.add_scalar(f'train/dsc_{cls}', dsc[1][cls], epoch)
            tb_logger.add_scalar(f'val_source/dsc_{cls}', source_dsc[1][cls], epoch)
            tb_logger.add_scalar(f'val_source/assd_{cls}', min(source_assd[1][cls], max_dist), epoch)
            tb_logger.add_scalar(f'val_target/dsc_{cls}', target_dsc[1][cls], epoch)
            tb_logger.add_scalar(f'val_target/assd_{cls}', min(target_assd[1][cls], max_dist), epoch)
        tb_logger.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_checkpoint(model, optimizer, opt, epoch, save_file)

    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_checkpoint(model, optimizer, opt, opt.epochs, save_file)


if __name__ == '__main__':
    # if torch.cuda.device_count() > 1:
    #     torch.multiprocessing.set_start_method('spawn')
    main()
