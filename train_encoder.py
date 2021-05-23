import os
import argparse
import time
import math

import monai
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from monai.data import DataLoader
from matplotlib import pyplot as plt
from tqdm import tqdm

from util import \
    AverageMeter, adjust_learning_rate, warmup_learning_rate, set_optimizer, save_checkpoint, \
    set_up_logger, log_parameters
from networks import get_encoder_model
from losses import SupConLoss
from data.dataset import CachePartDataset, CacheSliceDataset
from data.sampler import MultiDomainBatchSampler
from data.VS_SEG import get_encoder_train_transforms, get_encoder_val_transforms, get_data


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
    parser.add_argument('--batch_parts', type=int, default=None,
                        help='number of partitions in a batch (default: all partitions in volume)')
    parser.add_argument('--n_parts', type=int,
                        help='number of partitions in a volume')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--in_domain', type=int, default=100,
                        help='epochs with in-domain loss only')

    # model dataset
    parser.add_argument('--model', type=str,
                        choices=['SegResNet', 'ResUNet', 'ResUNet++'])
    parser.add_argument('--dataset', type=str, default='VS_SEG',
                        choices=['VS_SEG', 'crossMoDA'], help='dataset')
    parser.add_argument("--split", type=str, default="split.csv",
                        help="path to CSV file that defines training, validation and test datasets")
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    parser.add_argument('--size', type=int, default=384, help='expected size of input')

    # temperature
    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for loss function')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')

    opt = parser.parse_args()

    # set the path according to the environment
    if opt.data_folder is None:
        opt.data_folder = './datasets/'

    if opt.batch_parts is None:
        opt.batch_parts = opt.n_parts

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = f"{opt.dataset}_enc_{opt.model}_prts_{opt.n_parts}_lr_{opt.learning_rate}_decay_{opt.weight_decay}_" + \
                     f"bsz_{opt.batch_size}_bprts_{opt.batch_parts}_temp_{opt.temp}_indom_{opt.in_domain}"

    if opt.cosine:
        opt.model_name += "_cosine"

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name += "{}_warm"
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

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


def set_loader(opt):
    source_train_transform, target_train_transform = get_encoder_train_transforms(opt)
    source_val_transform, target_val_transform = get_encoder_val_transforms(opt)
    (source_train_data, target_train_data), (source_val_data, target_val_data), _ = get_data(opt)

    # train data
    if len(source_train_data) != len(target_train_data):
        raise ValueError(f"Length of source domain train data {(len(source_train_data))} is not equal to "
                         f"the length of target domain train data ({len(target_train_data)}).")
    source_train_dataset = CachePartDataset(data=source_train_data, transform=source_train_transform,
                                            parts_num=opt.n_parts, keys=["image"], num_workers=opt.num_workers)
    target_train_dataset = CachePartDataset(data=target_train_data, transform=target_train_transform,
                                            parts_num=opt.n_parts, keys=["image"], num_workers=opt.num_workers)
    train_dataset = ConcatDataset([source_train_dataset, target_train_dataset])
    batch_sampler = MultiDomainBatchSampler(
        domains_lengths=[len(source_train_data), len(target_train_data)],
        batch_size=opt.batch_size,
        parts_num=opt.n_parts,
        batch_parts_num=opt.batch_parts)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_sampler=batch_sampler,
        num_workers=opt.num_workers,
        pin_memory=True)

    # validation data
    source_val_dataset = CacheSliceDataset(data=source_val_data, transform=source_val_transform,
                                           keys=["image"], num_workers=opt.num_workers)
    target_val_dataset = CacheSliceDataset(data=target_val_data, transform=target_val_transform,
                                           keys=["image"], num_workers=opt.num_workers)
    source_val_loader = DataLoader(
        dataset=source_val_dataset,
        batch_size=math.ceil(opt.batch_size//2),
        num_workers=opt.num_workers//3,
        pin_memory=True)
    target_val_loader = DataLoader(
        dataset=target_val_dataset,
        batch_size=math.ceil(opt.batch_size//2),
        num_workers=opt.num_workers//3,
        pin_memory=True)

    return train_loader, source_val_loader, target_val_loader


def set_model(opt):
    model = get_encoder_model(opt)
    criterion = SupConLoss(temperature=opt.temp)

    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion


def check_data(train_loader, opt):
    logger = opt.logger
    samples_path = os.path.join(opt.log_folder, "samples")
    if not os.path.isdir(samples_path):
        os.makedirs(samples_path)
    first_batch = True
    logger.info("-" * 10)
    logger.info("Summary of the training data")
    logger.info(f"number of volumes    = {len(train_loader.dataset)}")
    logger.info(f"number of batches    = {len(train_loader)}")
    for idx, batch_data in enumerate(tqdm(train_loader, desc="Saving sampled data"), start=1):
        if first_batch:
            logger.info(f"data keys            = {list(batch_data.keys())}")
            image_orig = batch_data["image_orig"][0]
            image_views = tuple(batch_data["image"][0, :, ...])
            logger.info(f"original image shape = {tuple(image_orig.shape)}")
            logger.info(f"image views shapes   = {[tuple(v.shape) for v in image_views]}")
            first_batch = False
        total_bsz = batch_data["image"].shape[0]
        domain_bsz = batch_data["image"].shape[0] // 2
        for i in range(total_bsz):
            domain = ("source", "target")[i//domain_bsz]
            image_orig = batch_data["image_orig"][i, 0, ...]
            image_views = tuple(batch_data["image"][i, :, 0, ...])
            part_num = batch_data["part_num"][i]
            file_name = batch_data["image_meta_dict"]["filename_or_obj"][i]
            cols = len(image_views)+1
            fig = plt.figure("check", (cols*6, 6))
            fig.suptitle(f"Batch:{idx}  Item:{i+1}({domain})  Part:{part_num.item()+1}  File:{file_name}", fontsize=16)
            plt.subplot(1, cols, 1)
            plt.title("Original Image")
            plt.imshow(image_orig, cmap="gray", interpolation="none")
            for v_idx, image_view in enumerate(image_views, start=1):
                plt.subplot(1, cols, v_idx+1)
                plt.title(f"View_{v_idx}")
                plt.imshow(image_view.cpu(), cmap="gray", interpolation="none")
            plt.savefig(os.path.join(samples_path, f"batch_{idx}_item_{i+1}_{domain}_part_{part_num.item()+1}.png"))
    logger.info("-" * 10)


def train(train_loader, model, criterion, optimizer, epoch, iters, opt):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    partial_losses = {"source": AverageMeter(), "target": AverageMeter(), "cross": AverageMeter()}
    losses = AverageMeter()
    sources = AverageMeter()
    targets = AverageMeter()

    end = time.time()
    for idx, batch_data in enumerate(train_loader):
        data_time.update(time.time() - end)
        iters[0] += 1

        bsz = batch_data["image"].shape[0] // 2
        source_images = batch_data["image"][:bsz]
        source_labels = batch_data["part_num"][:bsz]
        target_images = batch_data["image"][bsz:]
        target_labels = batch_data["part_num"][bsz:]

        if torch.cuda.is_available():
            source_images = source_images.cuda(non_blocking=True)
            source_labels = source_labels.cuda(non_blocking=True)
            target_images = target_images.cuda(non_blocking=True)
            target_labels = target_labels.cuda(non_blocking=True)

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        s = source_images.shape
        source_images = source_images.reshape(s[0]*s[1], s[2], s[3], s[4])
        t = target_images.shape
        target_images = target_images.reshape(t[0]*t[1], t[2], t[3], t[4])

        # SOURCE DOMAIN
        source_features = model(source_images)
        source_features = source_features.reshape(s[0], s[1], -1)
        source_loss = criterion(source_features, source_labels)

        # TARGET DOMAIN
        target_features = model(target_images)
        target_features = target_features.reshape(t[0], t[1], -1)
        target_loss = criterion(target_features, target_labels)

        # CROSS-DOMAIN
        if epoch > opt.in_domain:
            all_features = torch.cat((source_features, target_features), dim=0)
            all_labels = torch.cat((source_labels, target_labels), dim=0)
            mask_all = torch.eq(all_labels, all_labels.T).float()
            mask_source = torch.eq(source_labels, source_labels.T).float()
            mask_target = torch.eq(target_labels, target_labels.T).float()
            mask_all[:bsz, :bsz] -= mask_source
            mask_all[bsz:, bsz:] -= mask_target
            cross_loss = criterion(all_features, mask=mask_all)
        else:
            cross_loss = 0

        # total loss
        loss = source_loss + target_loss + cross_loss

        # update metric
        partial_losses["source"].update(source_loss.item(), bsz)
        partial_losses["target"].update(target_loss.item(), bsz)
        partial_losses["cross"].update(0 if cross_loss == 0 else cross_loss.item(), bsz)
        losses.update(loss.item(), bsz)

        # keeping source and target features avg
        mean_source = torch.mean(source_features[:, 0, ...], dim=0)
        sources.update(mean_source, bsz)
        mean_target = torch.mean(target_features[:, 0, ...], dim=0)
        targets.update(mean_target, bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            # calculate current and average Normalized Mean Distance
            nmd = torch.dist(mean_source, mean_target, 2).item()
            nmd_avg = torch.dist(sources.avg, targets.avg, 2).item()
            opt.logger.info(f"Train: [{epoch}][{idx+1}/{len(train_loader)}]\t" +
                f"BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t" +
                f"DT {data_time.val:.3f} ({data_time.avg:.3f})\t" +
                f"loss {losses.val:.3f} <{', '.join(f'{pl.val:.3f}' for pl in partial_losses.values())}> " +
                f"({losses.avg:.3f} <{', '.join(f'{pl.avg:.3f}' for pl in partial_losses.values())}>)\t" +
                f"nmd {nmd:.3f} ({nmd_avg:.3f})")

    nmd_avg = torch.dist(sources.avg, targets.avg, 2).item()
    return losses.avg, {name: pl.avg for name, pl in partial_losses.items()}, nmd_avg


def validate(source_val_loader, target_val_loader, model):
    """validation"""
    model.eval()

    with torch.no_grad():
        source_features = []
        for batch_data in source_val_loader:
            images = batch_data["image"]
            if torch.cuda.is_available():
                images = images.cuda(non_blocking=True)
            features = model(images)
            source_features.append(features)
        source_features = torch.cat(source_features, dim=0)

        target_features = []
        for batch_data in target_val_loader:
            images = batch_data["image"]
            if torch.cuda.is_available():
                images = images.cuda(non_blocking=True)
            features = model(images)
            target_features.append(features)
        target_features = torch.cat(target_features, dim=0)

        mean_source = torch.mean(source_features, dim=0)
        mean_target = torch.mean(target_features, dim=0)
        nmd = torch.dist(mean_source, mean_target, 2).item()

        return nmd


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

    # build model and criterion
    model, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model)

    if opt.debug:
        check_data(train_loader, opt)

    iters = [0]
    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss, partial_losses, nmd = train(train_loader, model, criterion, optimizer, epoch, iters, opt)
        time2 = time.time()
        val_nmd = validate(source_val_loader, target_val_loader, model)
        time3 = time.time()
        logger.info(f"epoch {epoch},  train_time {(time2 - time1):.2f},  val_time {(time3 - time2):.2f},  " +
                    f"avg_loss {loss:.3f} <{', '.join(f'{pl:.3f}' for pl in partial_losses.values())}>,  "
                    f"avg_nmd {nmd:.3f},  val_nmd {val_nmd:.3f}")

        # tensorboard logging
        tb_logger.add_scalar('loss/total', loss, epoch)
        for name, p_loss in partial_losses.items():
            tb_logger.add_scalar(f'loss/{name}', p_loss, epoch)
        tb_logger.add_scalar('metric/nmd', nmd, epoch)
        tb_logger.add_scalar('metric/val_nmd', val_nmd, epoch)
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
