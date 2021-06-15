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

from metrics import compute_f1_score
from optimizers import get_optimizer
from util import \
    AverageMeter, adjust_learning_rate, warmup_learning_rate, save_checkpoint, set_up_logger, log_parameters
from networks import SegAuxModel
from losses import SupConLoss
from data.datasets import CachePartDataset, CacheSliceDataset
from data.samplers import MultiDomainBatchSampler
from data.VS_SEG import get_encoder_train_transforms, get_encoder_val_transforms, get_data, data_cfg


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
    parser.add_argument('--optimizer', type=str, default='SGD',
                        choices=['SGD', 'Adam'], help='optimizer')
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
                        choices=['SegResNet', 'ResUNet', 'ResUNet++', "DR-UNet104"])
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
    parser.add_argument('--classifier_eval', action='store_true',
                        help='adding two auxiliary linear classifiers to predict roi existence and partition index.')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')

    opt = parser.parse_args()

    if opt.data_folder is None:
        opt.data_folder = './datasets/'

    if opt.batch_parts is None:
        opt.batch_parts = opt.n_parts

    opt.in_channels = data_cfg["input_channels"]
    opt.classes_num = data_cfg["classes_num"]
    opt.classes = data_cfg["classes"]

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = f"{opt.dataset}_enc_{opt.model}_{opt.optimizer}_lr_{opt.learning_rate}_decay_{opt.weight_decay}" + \
                     f"_bsz_{opt.batch_size}_prts_{opt.n_parts}_bprts_{opt.batch_parts}_temp_{opt.temp}_indom_{opt.in_domain}"

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
    logger = opt.logger
    source_train_transform, target_train_transform = get_encoder_train_transforms(opt)
    source_val_transform, target_val_transform = get_encoder_val_transforms(opt)
    (source_train_data, target_train_data), (source_val_data, target_val_data), _ = get_data(opt)
    keys = ["image"] if not opt.classifier_eval else ["image", "label"]

    # train data
    if len(source_train_data) != len(target_train_data):
        raise ValueError(f"Length of source domain train data {(len(source_train_data))} is not equal to "
                         f"the length of target domain train data ({len(target_train_data)}).")
    logger.info("Caching source training data ...")
    source_train_dataset = CachePartDataset(data=source_train_data, transform=source_train_transform,
                                            parts_num=opt.n_parts, keys=keys, num_workers=1)
    logger.info("Caching target training data ...")
    target_train_dataset = CachePartDataset(data=target_train_data, transform=target_train_transform,
                                            parts_num=opt.n_parts, keys=keys, num_workers=1)
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
    logger.info(f"Summary of the training data:")
    logger.info(f"number of partitions = {len(train_dataset)}")
    logger.info(f"number of batches    = {len(train_loader)}")

    # validation data
    logger.info("Caching source validation data ...")
    source_val_dataset = CacheSliceDataset(data=source_val_data, transform=source_val_transform,
                                           keys=keys, num_workers=opt.num_workers//3)
    logger.info("Caching target validation data ...")
    target_val_dataset = CacheSliceDataset(data=target_val_data, transform=target_val_transform,
                                           keys=keys, num_workers=opt.num_workers//3)
    source_val_loader = DataLoader(
        dataset=source_val_dataset,
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        pin_memory=True)
    target_val_loader = DataLoader(
        dataset=target_val_dataset,
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        pin_memory=True)
    logger.info(f"Summary of the validation data:")
    logger.info(f"number of slices  (source/target) = {len(source_val_dataset)}/{len(target_val_dataset)}")
    logger.info(f"number of batches (source/target) = {len(source_val_loader)}/{len(target_val_loader)}")

    return train_loader, source_val_loader, target_val_loader


def set_model(opt):
    criteria = torch.nn.ModuleDict()
    if not opt.classifier_eval:
        model = SegAuxModel(
            mode="encoder+projector",
            base_args={
                "in_channels": opt.in_channels,
                "classes_num": opt.classes_num,
                "model": opt.model,
                "size": opt.size,
            }
        )
        criteria["ssl"] = SupConLoss(temperature=opt.temp)
    else:
        model = SegAuxModel(
            mode="encoder+projector+classifier",
            base_args={
                "in_channels": opt.in_channels,
                "classes_num": opt.classes_num,
                "model": opt.model,
                "size": opt.size,
            },
            enc_proj_args={"head": "mlp", "feat_dim": 128},
            classifiers_args={
                "outputs_sizes": [opt.n_parts, opt.classes_num-1],
                "detach": [True, True]
            }
        )
        criteria["ssl"] = SupConLoss(temperature=opt.temp)
        criteria["partitions"] = torch.nn.CrossEntropyLoss()
        criteria["classes"] = torch.nn.BCEWithLogitsLoss()

    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        model = model.cuda()
        criteria = criteria.cuda()
        cudnn.benchmark = True

    optimizer = get_optimizer(model.parameters(), opt)

    return model, criteria, optimizer


def check_train_data(train_loader, opt, max_samples=None):
    logger = opt.logger
    samples_path = os.path.join(opt.log_folder, "train_samples")
    if not os.path.isdir(samples_path):
        os.makedirs(samples_path)
    logger.info(f"Saving train sample data to {samples_path}")
    max_samples = len(train_loader.dataset) if max_samples is None else max_samples
    samples_num = 0
    for idx, batch_data in enumerate(tqdm(train_loader, desc="Saving  samples"), start=1):
        total_bsz = batch_data["image"].shape[0]
        domain_bsz = batch_data["image"].shape[0] // 2
        for i in range(total_bsz):
            samples_num += 1
            if samples_num > max_samples:
                continue
            domain = ("source", "target")[i//domain_bsz]
            image_orig = batch_data["image_orig"][i, 0, ...]
            image_views = tuple(batch_data["image"][i, :, 0, ...])
            part_idx = batch_data["part_idx"][i].item()
            file_name = batch_data["image_meta_dict"]["filename_or_obj"][i]
            cols = len(image_views)+1
            fig = plt.figure("check", (cols*6, 6))
            fig.suptitle(f"Batch:{idx}  Item:{i+1}({domain})  Part:{part_idx+1}  File:{file_name}", fontsize=16)
            plt.subplot(1, cols, 1)
            plt.title(f"Original Image {tuple(image_orig.shape)}")
            plt.imshow(image_orig, cmap="gray", interpolation="none")
            for v_idx, image_view in enumerate(image_views, start=1):
                plt.subplot(1, cols, v_idx+1)
                plt.title(f"View_{v_idx} {tuple(image_view.shape)}")
                plt.imshow(image_view, cmap="gray", interpolation="none")  # cpu() needed if Affine transform device set to cuda
            plt.savefig(os.path.join(samples_path, f"batch_{idx}_item_{i+1}_{domain}_part_{part_idx+1}.png"))


def check_val_data(data_loader, opt, name, max_samples=None):
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
            vol_idx = batch_data["vol_idx"][i].item()
            slice_idx = batch_data["slice_idx"][i].item()
            file_name = batch_data["image_meta_dict"]["filename_or_obj"][i]
            fig = plt.figure("check", (2*6, 6))
            fig.suptitle(f"Batch:{idx}  Item:{i+1}  Volume:{vol_idx+1} Slice:{slice_idx+1}  "
                         f"File:{file_name}", fontsize=16)
            plt.subplot(1, 2, 1)
            plt.title(f"Original Image {tuple(image_orig.shape)}")
            plt.imshow(image_orig, cmap="gray", interpolation="none")
            plt.subplot(1, 2, 2)
            plt.title(f"Image {tuple(image.shape)}")
            plt.imshow(image, cmap="gray", interpolation="none")
            plt.savefig(os.path.join(samples_path, f"batch_{idx}_item_{i+1}_vol_{vol_idx+1}_slice_{slice_idx+1}.png"))


def train(train_loader, model, criteria, optimizer, epoch, opt):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    cls_losses = AverageMeter()
    partial_losses = AverageMeter()
    losses = AverageMeter()
    sources = AverageMeter()
    targets = AverageMeter()

    end = time.time()
    for idx, batch_data in enumerate(train_loader):
        data_time.update(time.time() - end)

        bsz = batch_data["image"].shape[0] // 2
        source_images = batch_data["image"][:bsz]
        source_part_idxs = batch_data["part_idx"][:bsz]
        target_images = batch_data["image"][bsz:]
        target_part_idxs = batch_data["part_idx"][bsz:]

        if torch.cuda.is_available():
            source_images = source_images.cuda(non_blocking=True)
            source_part_idxs = source_part_idxs.cuda(non_blocking=True)
            target_images = target_images.cuda(non_blocking=True)
            target_part_idxs = target_part_idxs.cuda(non_blocking=True)

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss_ss
        s = source_images.shape
        source_images = source_images.reshape(s[0]*s[1], s[2], s[3], s[4])
        t = target_images.shape
        target_images = target_images.reshape(t[0]*t[1], t[2], t[3], t[4])

        # SOURCE DOMAIN
        source_output = model(source_images)
        source_features = source_output["encoder_projector"].reshape(s[0], s[1], -1)
        source_loss = criteria["ssl"](source_features, source_part_idxs)

        # TARGET DOMAIN
        target_output = model(target_images)
        target_features = target_output["encoder_projector"].reshape(t[0], t[1], -1)
        target_loss = criteria["ssl"](target_features, target_part_idxs)

        # CROSS-DOMAIN
        if epoch > opt.in_domain:
            all_features = torch.cat((source_features, target_features))
            all_prat_idxs = torch.cat((source_part_idxs, target_part_idxs))
            mask_all = torch.eq(all_prat_idxs, all_prat_idxs.T).float()
            # mask-out all in-domain positives
            mask_all[:bsz, :bsz] = 0
            mask_all[bsz:, bsz:] = 0
            neg_mask_all = torch.ones_like(mask_all)
            # mask-out all in-domain negatives
            neg_mask_all[:bsz, :bsz] = 0
            neg_mask_all[bsz:, bsz:] = 0
            cross_loss = criteria["ssl"](all_features, mask=mask_all, neg_mask=neg_mask_all)
        else:
            cross_loss = torch.tensor(0).to(source_loss.device)

        # CLASSIFIER
        if opt.classifier_eval:
            # all_labels = batch_data["label"]
            # all_labels = torch.repeat_interleave(all_labels, repeats=2, dim=0)  # there are 2 views for each image
            # if torch.cuda.is_available():
            #     all_labels = all_labels.cuda(non_blocking=True)
            # all_prat_idxs = torch.cat((source_part_idxs, target_part_idxs))
            # all_prat_idxs = torch.repeat_interleave(all_prat_idxs, repeats=2)  # no dim used to flatten output
            # all_classifier0_output = torch.cat((source_output["classifier0"], target_output["classifier0"]))
            # all_classifier1_output = torch.cat((source_output["classifier1"], target_output["classifier1"]))
            # part_idx_loss = criteria["partitions"](all_classifier0_output, all_prat_idxs)
            # label_loss = criteria["classes"](all_classifier1_output, all_labels)
            source_labels = batch_data["label"][:bsz]
            source_labels = torch.repeat_interleave(source_labels, repeats=2, dim=0)  # there are 2 views for each image
            if torch.cuda.is_available():
                source_labels = source_labels.cuda(non_blocking=True)
            source_part_idxs = torch.repeat_interleave(source_part_idxs, repeats=2)  # no dim used to flatten output
            part_idx_loss = criteria["partitions"](source_output["classifier0"], source_part_idxs)
            label_loss = criteria["classes"](source_output["classifier1"], source_labels)
        else:
            part_idx_loss = torch.tensor(0).to(source_loss.device)
            label_loss = torch.tensor(0).to(source_loss.device)

        # total loss
        loss = source_loss + target_loss + cross_loss
        total_loss = loss + part_idx_loss + label_loss

        # update metric
        cls_losses.update(torch.stack([part_idx_loss, label_loss]), bsz)
        partial_losses.update(torch.stack([source_loss, target_loss, cross_loss]), bsz)
        losses.update(loss.item(), bsz)

        # keeping source and target features avg
        with torch.no_grad():
            mean_source = torch.mean(source_features[:, 0, ...], dim=0)
            sources.update(mean_source, bsz)
            mean_target = torch.mean(target_features[:, 0, ...], dim=0)
            targets.update(mean_target, bsz)

        # SGD
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            # calculate current and average Normalized Mean Distance
            nmd = torch.dist(mean_source, mean_target, 2).item()
            nmd_avg = torch.dist(sources.avg, targets.avg, 2).item()
            cls_loss_info = \
                f"Loss-Cls {', '.join(f'{cl.item():.3f}' for cl in cls_losses.avg)}\t" if opt.classifier_eval else ""
            opt.logger.info(f"[{epoch}][{idx + 1}/{len(train_loader)}]\t" +
                            f"BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t" +
                            f"DT {data_time.val:.3f} ({data_time.avg:.3f})\t" +
                            f"Loss {losses.val:06.3f} <{', '.join(f'{pl.item():.3f}' for pl in partial_losses.val)}> " +
                            f"({losses.avg:06.3f} <{', '.join(f'{pl.item():.3f}' for pl in partial_losses.avg)}>)\t" +
                            cls_loss_info +
                            f"NMD {nmd:.3f} ({nmd_avg:.3f})")

    nmd_avg = torch.dist(sources.avg, targets.avg, 2).item()
    return losses.avg, \
           {name: pl.item() for name, pl in zip(["source", "target", "cross"], partial_losses.avg)}, \
           {name: cl.item() for name, cl in zip(["partitions", "classes"], cls_losses.avg)}, \
           nmd_avg


def validate(source_val_loader, target_val_loader, model, opt):
    """validation"""
    model.eval()

    source_features, source_cls0, source_gt0, source_cls1, source_gt1 = [], [], [], [], []
    target_features, target_cls0, target_gt0, target_cls1, target_gt1 = [], [], [], [], []

    items = ((source_val_loader, source_features, source_cls0, source_gt0, source_cls1, source_gt1),
             (target_val_loader, target_features, target_cls0, target_gt0, target_cls1, target_gt1))

    with torch.no_grad():
        for data_loader, features, cls0, gt0, cls1, gt1 in items:
            for batch_data in data_loader:
                images = batch_data["image"]
                if torch.cuda.is_available():
                    images = images.cuda(non_blocking=True)
                outputs = model(images)
                features.append(outputs["encoder_projector"])
                if opt.classifier_eval:
                    cls0.append(outputs["classifier0"])
                    cls1.append(outputs["classifier1"])
                    part_idx = batch_data["part_idx"]
                    label = batch_data["label"]
                    if torch.cuda.is_available():
                        part_idx = part_idx.cuda(non_blocking=True)
                        label = label.cuda(non_blocking=True)
                    gt0.append(part_idx)
                    gt1.append(label)

        mean_source = torch.mean(torch.cat(source_features), dim=0)
        mean_target = torch.mean(torch.cat(target_features), dim=0)
        nmd = torch.dist(mean_source, mean_target, 2).item()

        cls_scores = dict()
        cls_avg_score = 0
        if opt.classifier_eval:
            cls_scores = {
                "f1_source_partitions": compute_f1_score(torch.cat(source_cls0), torch.cat(source_gt0)),
                "f1_source_classes": compute_f1_score(torch.cat(source_cls1), torch.cat(source_gt1), multilabel=True),
                "f1_target_partitions": compute_f1_score(torch.cat(target_cls0), torch.cat(target_gt0)),
                "f1_target_classes": compute_f1_score(torch.cat(target_cls1), torch.cat(target_gt1), multilabel=True)
            }
            cls_avg_score = sum(cls_scores.values())/4

        return nmd, cls_scores, cls_avg_score


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
    model, criteria, optimizer = set_model(opt)

    if opt.debug:
        check_train_data(train_loader, opt, max_samples=200)
        check_val_data(source_val_loader, opt, "source_val", max_samples=30)
        check_val_data(target_val_loader, opt, "target_val", max_samples=30)

    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss, partial_losses, cls_losses, nmd = train(train_loader, model, criteria, optimizer, epoch, opt)
        time2 = time.time()
        val_nmd, cls_scores, cls_avg_score = validate(source_val_loader, target_val_loader, model, opt)
        time3 = time.time()
        if opt.classifier_eval:
            cls_loss_info = f"Loss-Cls {', '.join(f'{cl:.3f}' for cl in cls_losses.values())},  "
            cls_score_info = f"F1-Cls {cls_avg_score:.3f},  "
        else:
            cls_loss_info, cls_score_info = "", ""
            cls_losses, cls_scores = dict(), dict()
            cls_avg_score = None
        logger.info(f"Epoch {epoch}  Train: Time {(time2 - time1):.2f},  " +
                    f"Loss {loss:.3f} <{', '.join(f'{pl:.3f}' for pl in partial_losses.values())}>,  " +
                    cls_loss_info +
                    f"NMD {nmd:.3f}")
        logger.info(f"{' '*(len(str(epoch))+1)}  Validation: Time {(time3 - time2):.2f},  " +
                    cls_score_info +
                    f"NMD {val_nmd:.3f}")

        # tensorboard logging
        tb_logger.add_scalar('loss/total', loss, epoch)
        for name, p_loss in partial_losses.items():
            tb_logger.add_scalar(f'loss/{name}', p_loss, epoch)
        for name, c_loss in cls_losses.items():
            tb_logger.add_scalar(f'loss_cls/{name}', p_loss, epoch)
        tb_logger.add_scalar('metric/train_nmd', nmd, epoch)
        tb_logger.add_scalar('metric/val_nmd', val_nmd, epoch)
        if cls_avg_score is not None:
            tb_logger.add_scalar(f'metric/f1_avg', cls_avg_score, epoch)
        for name, score in cls_scores.items():
            tb_logger.add_scalar(f'metric/{name}', score, epoch)
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
