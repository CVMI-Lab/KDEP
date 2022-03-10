import os
import random
import time
import cv2, math
import numpy as np
import logging
import argparse
import subprocess

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
from torch.utils.data import Subset, Dataset
import torch.multiprocessing as mp
import torch.distributed as dist
import apex
from tensorboardX import SummaryWriter

from util import dataset, transform, config, augmentation, reader
from util.reader import DataReader
from util.util import AverageMeter, poly_learning_rate, intersectionAndUnionGPU, linear_learning_rate_with_warmup
from util.validate_full_size import test, cal_acc
from util.util import AverageMeter, intersectionAndUnion, check_makedirs, colorize

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


def find_free_port():
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Semantic Segmentation')
    parser.add_argument('--config', type=str, default='config/ade20k/ade20k_pspnet50.yaml', help='config file')
    parser.add_argument('--exp_name', type=str, default='ex0', help='experiment name')
    parser.add_argument('--ckpt_name', type=str, default='ckpt', help='ckpt name')
    parser.add_argument('opts', help='see config/ade20k/ade20k_pspnet50.yaml for all options', default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    cfg.exp_name = args.exp_name
    cfg.ckpt_name = args.ckpt_name
    return cfg


def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger


def worker_init_fn(worker_id):
    random.seed(args.manual_seed + worker_id)


def main_process():
    return not args.multiprocessing_distributed or (
            args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0)


def check(args):
    assert args.classes > 1
    assert args.zoom_factor in [1, 2, 4, 8]
    if args.arch == 'psp' and args.layers not in [18,50]:
        assert (args.train_h - 1) % 8 == 0 and (args.train_w - 1) % 8 == 0
    elif args.arch == 'psa':
        if args.compact:
            args.mask_h = (args.train_h - 1) // (8 * args.shrink_factor) + 1
            args.mask_w = (args.train_w - 1) // (8 * args.shrink_factor) + 1
        else:
            assert (args.mask_h is None and args.mask_w is None) or (
                    args.mask_h is not None and args.mask_w is not None)
            if args.mask_h is None and args.mask_w is None:
                args.mask_h = 2 * ((args.train_h - 1) // (8 * args.shrink_factor) + 1) - 1
                args.mask_w = 2 * ((args.train_w - 1) // (8 * args.shrink_factor) + 1) - 1
            else:
                assert (args.mask_h % 2 == 1) and (args.mask_h >= 3) and (
                        args.mask_h <= 2 * ((args.train_h - 1) // (8 * args.shrink_factor) + 1) - 1)
                assert (args.mask_w % 2 == 1) and (args.mask_w >= 3) and (
                        args.mask_w <= 2 * ((args.train_h - 1) // (8 * args.shrink_factor) + 1) - 1)
    elif args.arch == 'deeplabv2' or 'psptrans':
        pass
    else:
        raise Exception('architecture not supported yet'.format(args.arch))


def main():
    args = get_parser()
    check(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.train_gpu)
    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        cudnn.benchmark = False
        cudnn.deterministic = True
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.ngpus_per_node = len(args.train_gpu)
    if len(args.train_gpu) == 1:
        args.sync_bn = False
        args.distributed = False
        args.multiprocessing_distributed = False
    if args.multiprocessing_distributed:
        port = find_free_port()
        args.dist_url = f"tcp://127.0.0.1:{port}"
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args))
    else:
        main_worker(args.train_gpu, args.ngpus_per_node, args)


def local_data_prepare():
    args.data_root = os.path.join(args.local_prefix, args.data_root)
    args.train_list = os.path.join(args.local_prefix, args.train_list)
    args.val_list = os.path.join(args.local_prefix, args.val_list)
    args.test_list = os.path.join(args.local_prefix, args.test_list)
    cmd_line = "mkdir -p {0}".format(args.save_folder + '/gray')
    subprocess.call(cmd_line.split())


def main_worker(gpu, ngpus_per_node, argss):
    global args
    args = argss

    args.save_path = args.save_path + args.exp_name + '/model'
    args.save_folder = args.save_folder + args.exp_name + '/result/val'
    args.initpath = args.ckpt_name

    local_data_prepare()

    if args.sync_bn:
        if args.multiprocessing_distributed:
            BatchNorm = apex.parallel.SyncBatchNorm
        else:
            from lib.sync_bn.modules import BatchNorm2d
            BatchNorm = BatchNorm2d
    else:
        BatchNorm = nn.BatchNorm2d
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size,
                                rank=args.rank)

    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)
    args.index_split = 5

    if args.arch == 'psp':
        from model.pspnet import PSPNet
        # from model.psptrans import PSPNet
        model = PSPNet(layers=args.layers, classes=args.classes, zoom_factor=args.zoom_factor, criterion=criterion,
                       BatchNorm=BatchNorm, initpath=args.initpath)
        modules_ori = [model.layer0, model.layer1, model.layer2, model.layer3, model.layer4]
        modules_new = [model.ppm, model.cls, model.aux]
    elif args.arch == 'mobilev2':
        from model.pspnet import Mobilev2_PSP
        # from model.psptrans import PSPNet
        model = Mobilev2_PSP(layers=args.layers, classes=args.classes, zoom_factor=args.zoom_factor, criterion=criterion,
                       BatchNorm=BatchNorm, initpath=args.initpath)
        modules_ori = [model.features, model.conv]
        modules_new = [model.ppm, model.cls]
        args.index_split = 2
    elif args.arch == 'psptrans':
        # from model.pspnet import PSPNet
        from model.psptrans import PSPNet
        model = PSPNet(layers=args.layers, classes=args.classes, zoom_factor=args.zoom_factor, criterion=criterion,
                       BatchNorm=BatchNorm)
        # modules_ori = [model.layer0, model.layer1, model.layer2, model.layer3, model.layer4]
        # modules_ori = [model.layer0, model.layer1, model.layer2, model.layer3]
        modules_ori = [model.layer0, model.layer1, model.layer2, model.layer3,model.rpn_tokens, model.conv]
        modules_new = [model.ppm, model.cls, model.aux]
    elif args.arch == 'psa':
        from model.psanet import PSANet
        model = PSANet(layers=args.layers, classes=args.classes, zoom_factor=args.zoom_factor, psa_type=args.psa_type,
                       compact=args.compact, shrink_factor=args.shrink_factor, mask_h=args.mask_h, mask_w=args.mask_w,
                       normalization_factor=args.normalization_factor, psa_softmax=args.psa_softmax,
                       criterion=criterion,
                       BatchNorm=BatchNorm)
        modules_ori = [model.layer0, model.layer1, model.layer2, model.layer3, model.layer4]
        modules_new = [model.psa, model.cls, model.aux]

    elif args.arch == 'deeplabv2':
        from model.deeplabv2 import Resnet101_deeplab
        print("args.pretrain data=" + args.pretrain_data)
        # import ipdb; ipdb.set_trace(context=20)
        model = Resnet101_deeplab(num_classes=args.classes, criterion=criterion, pretrained=True,
                                  pretrain_data=args.pretrain_data)
        modules_ori = model.pretrained_layers()
        modules_new = model.new_layers()

    params_list = []
    for module in modules_ori:
        # print(module)
        if isinstance(module, nn.Module):
            params_list.append(dict(params=module.parameters(), lr=args.base_lr))
        else:
            params_list.append(dict(params=module, lr=args.base_lr))
    for module in modules_new:
        params_list.append(dict(params=module.parameters(), lr=args.base_lr * 10))

    if args.optim_type == 'sgd':
        optimizer = torch.optim.SGD(params_list, lr=args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optim_type == 'adamw':
        optimizer = torch.optim.AdamW(params_list, eps=1e-8, betas=(0.9, 0.999),
                                lr=args.base_lr, weight_decay=args.weight_decay)
    else:
        raise Exception('optimizer not supported yet')

    global logger, writer
    logger = get_logger()
    writer = SummaryWriter(args.save_path)
    logger.info(args)

    # --- prepare KD teacher model
    if args.teacher_weight:
        logger.info("=> creating Teacher model ...")
        from model.pspnet import PSPNet
        # from model.psptrans import PSPNet
        model_teacher = PSPNet(layers=args.teacher_layers, classes=args.classes, zoom_factor=args.zoom_factor, criterion=criterion,
                       BatchNorm=BatchNorm, initpath=args.initpath)

        if args.distributed:
            torch.cuda.set_device(gpu)
            if args.use_apex:
                import copy
                optimizer_t = copy.deepcopy(optimizer)
                model_teacher, optimizer_ = apex.amp.initialize(model_teacher.cuda(), optimizer_t,
                                                               opt_level=args.opt_level,
                                                               keep_batchnorm_fp32=args.keep_batchnorm_fp32,
                                                               loss_scale=args.loss_scale)
                model_teacher = apex.parallel.DistributedDataParallel(model_teacher)
            else:
                model_teacher = torch.nn.parallel.DistributedDataParallel(model_teacher.cuda(), device_ids=[gpu])
        else:
            model_teacher = torch.nn.DataParallel(model_teacher.cuda())

        if os.path.isfile(args.teacher_weight):
            if main_process():
                logger.info("=> loading Teacher weight '{}'".format(args.teacher_weight))
            checkpoint = torch.load(args.teacher_weight)
            model_teacher.load_state_dict(checkpoint['state_dict'])
            if main_process():
                logger.info("=> loaded Teacher weight '{}'".format(args.teacher_weight))
        else:
            if main_process():
                logger.info("=> no weight found at '{}'".format(args.teacher_weight))


    logger.info("=> creating model ...")
    logger.info("Classes: {}".format(args.classes))
    if args.distributed:
        torch.cuda.set_device(gpu)
        args.batch_size = int(args.batch_size / ngpus_per_node)
        args.unlabelled_batch_size = int(args.unlabelled_batch_size / ngpus_per_node)
        args.batch_size_val = int(args.batch_size_val / ngpus_per_node)
        args.workers = int(args.workers / ngpus_per_node)
        if args.use_apex:
            model, optimizer = apex.amp.initialize(model.cuda(), optimizer, opt_level=args.opt_level,
                                                   keep_batchnorm_fp32=args.keep_batchnorm_fp32,
                                                   loss_scale=args.loss_scale)
            model = apex.parallel.DistributedDataParallel(model)
        else:
            model = torch.nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[gpu])

    else:
        model = torch.nn.DataParallel(model.cuda())

    if args.weight:
        if os.path.isfile(args.weight):
            if main_process():
                logger.info("=> loading weight '{}'".format(args.weight))
            checkpoint = torch.load(args.weight)
            model.load_state_dict(checkpoint['state_dict'])
            if main_process():
                logger.info("=> loaded weight '{}'".format(args.weight))
        else:
            if main_process():
                logger.info("=> no weight found at '{}'".format(args.weight))

    if args.resume:
        if os.path.isfile(args.resume):
            if main_process():
                logger.info("=> loading checkpoint '{}'".format(args.resume))
            # checkpoint = torch.load(args.resume)
            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda())
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            if main_process():
                logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            if main_process():
                logger.info("=> no checkpoint found at '{}'".format(args.resume))

    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]
    train_labelled_loader, train_labelled_sampler = get_labeled_unlabeled_pseudo_dataloader(args)

    if args.evaluate:
        if args.evaluate_full_size is False:
            val_transform = transform.Compose([
                transform.Crop([args.train_h, args.train_w], crop_type='center', padding=mean,
                               ignore_label=args.ignore_label),
                transform.ToTensor(),
                transform.Normalize(mean=mean, std=std)])
            if args.data_root[-6:] =='ade20k':
                val_data = dataset.SemData_ade20k(split='val', data_root=args.data_root, data_list=args.val_list,
                                           transform=val_transform)
            else:
                val_data = dataset.SemData(split='val', data_root=args.data_root, data_list=args.val_list,
                                           transform=val_transform)
            if args.distributed:
                val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)
            else:
                val_sampler = None
            val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size_val, shuffle=False,
                                                     num_workers=args.workers, pin_memory=True, sampler=val_sampler)

        else:
            gray_folder = os.path.join(args.save_folder, 'gray')
            color_folder = os.path.join(args.save_folder, 'color')
            val_transform = transform.Compose([transform.ToTensor()])
            if args.data_root[-6:] == 'ade20k':
                val_data = dataset.SemData_ade20k(split=args.split, data_root=args.data_root, data_list=args.val_list,
                                           transform=val_transform)
            else:
                val_data = dataset.SemData(split=args.split, data_root=args.data_root, data_list=args.val_list,
                                       transform=val_transform)

            val_loader = torch.utils.data.DataLoader(val_data, batch_size=1, shuffle=False, num_workers=args.workers,
                                                     pin_memory=True)
            colors = np.loadtxt(args.colors_path).astype('uint8')
            names = [line.rstrip('\n') for line in open(args.names_path)]

            if args.arch == 'psp':
                model_val = PSPNet(layers=args.layers, classes=args.classes, zoom_factor=args.zoom_factor,
                                   pretrained=False)
            elif args.arch == 'mobilev2':
                model_val = Mobilev2_PSP(layers=args.layers, classes=args.classes, zoom_factor=args.zoom_factor)
            elif args.arch == 'psptrans':
                model_val = PSPNet(layers=args.layers, classes=args.classes, zoom_factor=args.zoom_factor,
                               criterion=criterion,
                               BatchNorm=BatchNorm,pretrained=False)
            elif args.arch == 'deeplabv2':
                model_val = Resnet101_deeplab(num_classes=args.classes, criterion=criterion, pretrained=True,
                                              pretrain_data=args.pretrain_data)

    # ----- init save best ckpt vars
    best_val_miou = args.evaluate_previous_best_val_mIou
    best_ckpt_name = None

    # ---- for the file exist issue at slurm
    check_makedirs(gray_folder)
    # ---- for the file exist issue at slurm

    for epoch in range(args.start_epoch, args.epochs):
        epoch_log = epoch + 1

        if args.distributed:
            train_labelled_sampler.set_epoch(epoch)

        try:
            writer
        except NameError:
            writer = None

        try:
            list_path
        except NameError:
            list_path = None

        if args.teacher_weight:
            main_loss_train, total_loss_train, mIoU_train, mAcc_train, allAcc_train = \
                train_KD(train_labelled_loader, model, model_teacher, optimizer, epoch, args, writer, logger)
        else:
            main_loss_train, total_loss_train, mIoU_train, mAcc_train, allAcc_train= \
                train(train_labelled_loader, model, optimizer, epoch, args, writer, logger)

        if main_process():
            writer.add_scalar('main_loss_train', main_loss_train, epoch_log)
            writer.add_scalar('total_loss_train', total_loss_train, epoch_log)
            writer.add_scalar('mIoU_train', mIoU_train, epoch_log)
            writer.add_scalar('mAcc_train', mAcc_train, epoch_log)
            writer.add_scalar('allAcc_train', allAcc_train, epoch_log)

        # if (epoch_log % args.save_freq == 0) and main_process():
        #     filename = args.save_path + '/train_epoch_' + str(epoch_log) + '.pth'
        #     logger.info('Saving checkpoint to: ' + filename)
        #     torch.save({'epoch': epoch_log, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
        #                filename)

        if args.evaluate and (epoch_log > args.evaluate_start) and (
                (epoch_log - args.evaluate_start) % args.evaluate_freq == 0 or epoch_log == args.epochs) and main_process():
            if args.evaluate_full_size is False:
                loss_val, mIoU_val, mAcc_val, allAcc_val = validate(val_loader, model, criterion)
            else:
                with_module_dict = model.state_dict()
                for key in list(with_module_dict.keys()):
                    if 'module.' in key:
                        with_module_dict[key.replace('module.', '')] = with_module_dict[key]
                        del with_module_dict[key]
                model_val.load_state_dict(with_module_dict)
                model_val = model_val.cuda()
                gray_folder_ = os.path.join(gray_folder, str(epoch_log))
                mIoU_val, mAcc_val, allAcc_val = test(val_loader, val_data.data_list, model_val, args.classes, mean,
                                                      std,
                                                      args.base_size, args.test_h,
                                                      args.test_w, args.scales, gray_folder_, color_folder, colors,
                                                      names, args)
                loss_val = 0
            # -------- save best val mIou ckpt
            if mIoU_val > best_val_miou and main_process():
                if best_ckpt_name is not None:
                    if os.path.exists(best_ckpt_name):
                        os.remove(best_ckpt_name)
                        logger.info('Remove checkpoint: ' + best_ckpt_name)
                best_val_miou = mIoU_val
                filename = args.save_path + '/train_epoch_' + str(epoch_log) + '.pth'
                if not os.path.exists(filename):
                    logger.info('Saving checkpoint to: ' + filename)
                    torch.save(
                        {'epoch': epoch_log, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
                        filename)
                best_ckpt_name = filename

            elif mIoU_val <= best_val_miou and main_process():
                filename = args.save_path + '/train_epoch_' + str(epoch_log) + '.pth'
                if not os.path.exists(filename):
                    logger.info('Saving checkpoint to: ' + filename)
                    torch.save(
                        {'epoch': epoch_log, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
                        filename)

            if main_process():
                writer.add_scalar('loss_val', loss_val, epoch_log)
                writer.add_scalar('mIoU_val', mIoU_val, epoch_log)
                writer.add_scalar('mAcc_val', mAcc_val, epoch_log)
                writer.add_scalar('allAcc_val', allAcc_val, epoch_log)

    if main_process():
        logger.info("finish " + args.save_path)
        # os.system('hdfs dfs -put exp/* hdfs://haruna/home/byte_arnold_hl_vc/user/ruifeihe/exp/')
        # logger.info("putting exp/* to hdfs://haruna/home/byte_arnold_hl_vc/user/ruifeihe/exp/")


def get_labeled_unlabeled_pseudo_dataloader(args):
    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]

    train_transform = transform.Compose([
        # train_transform = Compose([
        transform.RandScale([args.scale_min, args.scale_max]),
        transform.RandRotate([args.rotate_min, args.rotate_max], padding=mean, ignore_label=args.ignore_label),
        transform.RandomGaussianBlur(),
        transform.PhotoMetricDistortion(),
        transform.RandomHorizontalFlip(),
        transform.Crop([args.train_h, args.train_w], crop_type='rand', padding=mean, ignore_label=args.ignore_label),
        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std)])

    if args.data_root[-6:] == 'ade20k':
        train_labelled_ds = dataset.SemData_ade20k(split='train', data_root=args.data_root, data_list=args.train_list,
                                            transform=train_transform)
    else:
        train_labelled_ds = dataset.SemData(split='train', data_root=args.data_root, data_list=args.train_list,
                                        transform=train_transform)


    print("len(train_labelled_ds)=" + str(len(train_labelled_ds)))

    if args.distributed:
        train_labelled_sampler = torch.utils.data.distributed.DistributedSampler(train_labelled_ds)
    else:
        train_labelled_sampler = None
    train_labelled_loader = torch.utils.data.DataLoader(train_labelled_ds, batch_size=args.batch_size,
                                                        shuffle=(train_labelled_sampler is None),
                                                        num_workers=args.workers, pin_memory=True,
                                                        sampler=train_labelled_sampler,
                                                        drop_last=True)

    return train_labelled_loader, train_labelled_sampler

def train(train_labelled_loader, model, optimizer, epoch, args, writer, logger):
    '''

    :param train_labelled_loader:
    :param model:
    :param optimizer:
    :param epoch:
    :param args:
    :param writer:
    :param logger:
    :return: all metrics
    '''
    batch_time = AverageMeter()
    data_time = AverageMeter()
    main_loss_meter = AverageMeter()
    aux_loss_meter = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    model.train()
    end = time.time()
    max_iter = args.epochs * len(train_labelled_loader)
    start_iter = args.start_epoch * len(train_labelled_loader)

    for i, (input, target) in enumerate(train_labelled_loader):

        data_time.update(time.time() - end)
        current_iter = epoch * len(train_labelled_loader) + i + 1
        if args.zoom_factor != 8:
            h = int((target.size()[1] - 1) / 8 * args.zoom_factor + 1)
            w = int((target.size()[2] - 1) / 8 * args.zoom_factor + 1)
            # 'nearest' mode doesn't support align_corners mode and 'bilinear' mode is fine for downsampling
            target = F.interpolate(target.unsqueeze(1).float(), size=(h, w), mode='bilinear',
                                   align_corners=True).squeeze(1).long()

        input = input.cuda(non_blocking=True)  # input.shape= Bx3xHxW
        target = target.cuda(non_blocking=True)  # TARGET.shape= BxHxW

        # ------- supervised
        output_pred, main_loss, aux_loss = model(input, target, sup_loss_method=args.sup_loss_method)
        output_pred_x, output_pred_aux = output_pred
        output = output_pred_x.max(1)[1]

        if not args.multiprocessing_distributed:
            main_loss, aux_loss = torch.mean(main_loss), torch.mean(aux_loss)
        loss = main_loss + args.aux_weight * aux_loss

        optimizer.zero_grad()
        if args.use_apex and args.multiprocessing_distributed:
            with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        optimizer.step()

        n = input.size(0)
        if args.multiprocessing_distributed:
            main_loss, aux_loss, loss = main_loss.detach() * n, aux_loss * n, loss * n  # not considering ignore pixels
            count = target.new_tensor([n], dtype=torch.long)
            dist.all_reduce(main_loss), dist.all_reduce(aux_loss), dist.all_reduce(loss), dist.all_reduce(
                count)
            n = count.item()
            main_loss, aux_loss, loss = main_loss / n, aux_loss / n, loss / n

        # get mIou for supervised part
        intersection, union, target = intersectionAndUnionGPU(output, target, args.classes, args.ignore_label)
        if args.multiprocessing_distributed:
            dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        main_loss_meter.update(main_loss.item(), n)
        aux_loss_meter.update(aux_loss.item(), n)
        loss_meter.update(loss.item(), n)
        batch_time.update(time.time() - end)
        end = time.time()

        if args.optim_type == 'sgd':
            current_lr = poly_learning_rate(args.base_lr, current_iter, max_iter, power=args.power)
        elif args.optim_type == 'adamw':
            current_lr = linear_learning_rate_with_warmup(args.base_lr, current_iter, max_iter, args, epoch, warmup_epochs=args.warmup_epochs)
        else:
            raise Exception('optimizer not supported yet')

        # logger.info('current lr = {}'.format(current_lr))
        for index in range(0, args.index_split):  # index_split=5
            optimizer.param_groups[index]['lr'] = current_lr
        for index in range(args.index_split, len(optimizer.param_groups)):
            optimizer.param_groups[index]['lr'] = current_lr * 10
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        if (i + 1) % args.print_freq == 0 and main_process() :
            logger.info('Epoch: [{}/{}][{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Remain {remain_time} '
                        'MainLoss {main_loss_meter.val:.4f} '
                        'AuxLoss {aux_loss_meter.val:.4f} '
                        'Loss {loss_meter.val:.4f} '
                        'Accuracy {accuracy:.4f}.'.format(epoch + 1, args.epochs, i + 1, len(train_labelled_loader),
                                                          batch_time=batch_time,
                                                          data_time=data_time,
                                                          remain_time=remain_time,
                                                          main_loss_meter=main_loss_meter,
                                                          aux_loss_meter=aux_loss_meter,
                                                          loss_meter=loss_meter,
                                                          accuracy=accuracy))
        if main_process() and writer is not None:
            writer.add_scalar('total_loss_train_batch', loss_meter.val, current_iter)
            writer.add_scalar('mIoU_train_batch', np.mean(intersection / (union + 1e-10)), current_iter)
            writer.add_scalar('mAcc_train_batch', np.mean(intersection / (target + 1e-10)), current_iter)
            writer.add_scalar('current_lr', current_lr, current_iter)

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    if main_process():
        logger.info(
            'Train result at epoch [{}/{}]: mIoU/mAcc/allAcc/loss {:.4f}/{:.4f}/{:.4f}/{:.4f}.'.format(
                epoch + 1, args.epochs, mIoU,
                mAcc, allAcc, loss_meter.avg))
    return main_loss_meter.avg, loss_meter.avg, mIoU, mAcc, allAcc

def train_KD(train_labelled_loader, model, model_teacher, optimizer, epoch, args, writer, logger):
    '''

    :param train_labelled_loader:
    :param model:
    :param optimizer:
    :param epoch:
    :param args:
    :param writer:
    :param logger:
    :return: all metrics
    '''
    batch_time = AverageMeter()
    data_time = AverageMeter()
    main_loss_meter = AverageMeter()
    aux_loss_meter = AverageMeter()
    loss_meter = AverageMeter()
    loss_KD_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    model.train()
    model_teacher.eval()
    end = time.time()
    max_iter = args.epochs * len(train_labelled_loader)
    start_iter = args.start_epoch * len(train_labelled_loader)

    for i, (input, target) in enumerate(train_labelled_loader):

        data_time.update(time.time() - end)
        current_iter = epoch * len(train_labelled_loader) + i + 1
        if args.zoom_factor != 8:
            h = int((target.size()[1] - 1) / 8 * args.zoom_factor + 1)
            w = int((target.size()[2] - 1) / 8 * args.zoom_factor + 1)
            # 'nearest' mode doesn't support align_corners mode and 'bilinear' mode is fine for downsampling
            target = F.interpolate(target.unsqueeze(1).float(), size=(h, w), mode='bilinear',
                                   align_corners=True).squeeze(1).long()

        input = input.cuda(non_blocking=True)  # input.shape= Bx3xHxW
        target = target.cuda(non_blocking=True)  # TARGET.shape= BxHxW

        # ------- supervised
        output_pred, main_loss, aux_loss = model(input, target, sup_loss_method=args.sup_loss_method)
        output_pred_x, output_pred_aux = output_pred
        output = output_pred_x.max(1)[1]
        logits_student = output_pred_x

        logits_teacher = model_teacher(input, sup_loss_method=args.sup_loss_method)

        class DistillKL(nn.Module):
            """Distilling the Knowledge in a Neural Network"""

            def __init__(self, T):
                super(DistillKL, self).__init__()
                self.T = T

            def forward(self, y_s, y_t):
                p_s = F.log_softmax(y_s / self.T, dim=1)
                p_t = F.softmax(y_t / self.T, dim=1)
                loss = F.kl_div(p_s, p_t, reduction='mean') * (self.T ** 2) / y_s.shape[0]
                return loss

        KD_criterion = DistillKL(args.KD_T)
        loss_KD = KD_criterion(logits_student, logits_teacher)

        # import ipdb
        # ipdb.set_trace(context=20)

        if not args.multiprocessing_distributed:
            main_loss, aux_loss = torch.mean(main_loss), torch.mean(aux_loss)
        supervised_loss = main_loss + args.aux_weight * aux_loss

        loss = (1-args.KD_a) * supervised_loss + args.KD_a * loss_KD

        optimizer.zero_grad()
        if args.use_apex and args.multiprocessing_distributed:
            with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        optimizer.step()

        n = input.size(0)
        if args.multiprocessing_distributed:
            main_loss, aux_loss, loss, loss_KD = main_loss.detach() * n, aux_loss * n, loss * n, loss_KD * n  # not considering ignore pixels
            count = target.new_tensor([n], dtype=torch.long)
            dist.all_reduce(main_loss), dist.all_reduce(aux_loss), dist.all_reduce(loss),dist.all_reduce(loss_KD), dist.all_reduce(
                count)
            n = count.item()
            main_loss, aux_loss, loss, loss_KD = main_loss / n, aux_loss / n, loss / n, loss_KD / n

        # get mIou for supervised part
        intersection, union, target = intersectionAndUnionGPU(output, target, args.classes, args.ignore_label)
        if args.multiprocessing_distributed:
            dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        main_loss_meter.update(main_loss.item(), n)
        aux_loss_meter.update(aux_loss.item(), n)
        loss_meter.update(loss.item(), n)
        loss_KD_meter.update(loss_KD.item(), n)
        batch_time.update(time.time() - end)
        end = time.time()

        if args.optim_type == 'sgd':
            current_lr = poly_learning_rate(args.base_lr, current_iter, max_iter, power=args.power)
        elif args.optim_type == 'adamw':
            current_lr = linear_learning_rate_with_warmup(args.base_lr, current_iter, max_iter, args, epoch, warmup_epochs=args.warmup_epochs)
        else:
            raise Exception('optimizer not supported yet')

        # logger.info('current lr = {}'.format(current_lr))
        for index in range(0, args.index_split):  # index_split=5
            optimizer.param_groups[index]['lr'] = current_lr
        for index in range(args.index_split, len(optimizer.param_groups)):
            optimizer.param_groups[index]['lr'] = current_lr * 10
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        if (i + 1) % args.print_freq == 0 and main_process() :
            logger.info('Epoch: [{}/{}][{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Remain {remain_time} '
                        'MainLoss {main_loss_meter.val:.4f} '
                        'AuxLoss {aux_loss_meter.val:.4f} '
                        'Loss_KD {loss_KD_meter.val:.4f} '
                        'Loss {loss_meter.val:.4f} '
                        'Accuracy {accuracy:.4f}.'.format(epoch + 1, args.epochs, i + 1, len(train_labelled_loader),
                                                          batch_time=batch_time,
                                                          data_time=data_time,
                                                          remain_time=remain_time,
                                                          main_loss_meter=main_loss_meter,
                                                          aux_loss_meter=aux_loss_meter,
                                                          loss_KD_meter=loss_KD_meter,
                                                          loss_meter=loss_meter,
                                                          accuracy=accuracy))
        if main_process() and writer is not None:
            writer.add_scalar('total_loss_train_batch', loss_meter.val, current_iter)
            writer.add_scalar('mIoU_train_batch', np.mean(intersection / (union + 1e-10)), current_iter)
            writer.add_scalar('mAcc_train_batch', np.mean(intersection / (target + 1e-10)), current_iter)
            writer.add_scalar('current_lr', current_lr, current_iter)

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    if main_process():
        logger.info(
            'Train result at epoch [{}/{}]: mIoU/mAcc/allAcc/loss {:.4f}/{:.4f}/{:.4f}/{:.4f}.'.format(
                epoch + 1, args.epochs, mIoU,
                mAcc, allAcc, loss_meter.avg))
    return main_loss_meter.avg, loss_meter.avg, mIoU, mAcc, allAcc


def validate(val_loader, model, criterion):
    if main_process():
        logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    model.eval()
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        data_time.update(time.time() - end)
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        output = model(input)
        if args.zoom_factor != 8:
            output = F.interpolate(output, size=target.size()[1:], mode='bilinear', align_corners=True)
        loss = criterion(output, target)

        n = input.size(0)
        if args.multiprocessing_distributed:
            loss = loss * n  # not considering ignore pixels
            count = target.new_tensor([n], dtype=torch.long)
            dist.all_reduce(loss), dist.all_reduce(count)
            n = count.item()
            loss = loss / n
        else:
            loss = torch.mean(loss)

        output = output.max(1)[1]
        intersection, union, target = intersectionAndUnionGPU(output, target, args.classes, args.ignore_label)
        if args.multiprocessing_distributed:
            dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        loss_meter.update(loss.item(), input.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        if ((i + 1) % args.print_freq == 0) and main_process():
            logger.info('Test: [{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                        'Accuracy {accuracy:.4f}.'.format(i + 1, len(val_loader),
                                                          data_time=data_time,
                                                          batch_time=batch_time,
                                                          loss_meter=loss_meter,
                                                          accuracy=accuracy))

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    if main_process():
        logger.info('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
        for i in range(args.classes):
            logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class[i], accuracy_class[i]))
        logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')
    return loss_meter.avg, mIoU, mAcc, allAcc


if __name__ == '__main__':
    main()

