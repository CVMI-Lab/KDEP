import argparse
import os

import cox.store
import numpy as np
import torch as ch
from cox import utils
# from robustness import datasets, defaults, model_utils, train
from robustness import datasets, defaults, model_utils
import train
import train_distill
from robustness.tools import helpers
from torch import nn
from torchvision import models
import torch
import models_rw
from utils import constants as cs
from utils import fine_tunify, transfer_datasets
import resnet, mobilenet

parser = argparse.ArgumentParser(description='Transfer learning via pretrained Imagenet models',
                                 conflict_handler='resolve')
parser = defaults.add_args_to_parser(defaults.CONFIG_ARGS, parser)
parser = defaults.add_args_to_parser(defaults.MODEL_LOADER_ARGS, parser)
parser = defaults.add_args_to_parser(defaults.TRAINING_ARGS, parser)
parser = defaults.add_args_to_parser(defaults.PGD_ARGS, parser)

# Custom arguments
parser.add_argument('--dataset', type=str, default='cifar',
                    help='Dataset (Overrides the one in robustness.defaults)')
parser.add_argument('--model-path', type=str, default='')
parser.add_argument('--resume', action='store_true',
                    help='Whether to resume or not (Overrides the one in robustness.defaults)')
parser.add_argument('--pytorch-pretrained', action='store_true',
                    help='If True, loads a Pytorch pretrained model.')
parser.add_argument('--cifar10-cifar10', action='store_true',
                    help='cifar10 to cifar10 transfer')
parser.add_argument('--subset', type=int, default=None,
                    help='number of training data to use from the dataset')
parser.add_argument('--no-tqdm', type=int, default=1,
                    choices=[0, 1], help='Do not use tqdm.')
parser.add_argument('--no-replace-last-layer', action='store_true',
                    help='Whether to avoid replacing the last layer')
parser.add_argument('--freeze-level', type=int, default=-1,
                    help='Up to what layer to freeze in the pretrained model (assumes a resnet architectures)')
parser.add_argument('--additional-hidden', type=int, default=0,
                    help='How many hidden layers to add on top of pretrained network + classification layer')
parser.add_argument('--per-class-accuracy', action='store_true', help='Report the per-class accuracy. '
                    'Can be used only with pets, caltech101, caltech256, aircraft, and flowers.')

# add
parser.add_argument('--weight', action='store_true',
                    help='Whether to put weights or not')
parser.add_argument('--weight_path', type=str, default=None, help='weight path')
parser.add_argument('--abandom_fc', action='store_true', default=False,
                    help='Whether abandom fc after leaned mask to ft')

# scheduler
parser.add_argument('--max_lr', type=float, default=1.0, help='1cycle.')
parser.add_argument('--custom_lr_schedule', type=str, default=None, help='custom_lr_schedule.')

# added for Knowledge Distillation

parser.add_argument('--KD_T', type=float, default=4, help='Temperature in Knowledge Distillation.')
parser.add_argument('--KD_T_mid', type=float, default=4, help='Temperature in Knowledge Distillation.')
parser.add_argument('--cluster_KD_T', type=float, default=4, help='Temperature in Knowledge Distillation.')
parser.add_argument('--KD_a', type=float, default=1.0, help='Balancing weight between losses in KD.')
parser.add_argument('--loss_w', type=float, default=0.0, help='Balancing weight between losses in KD.')
parser.add_argument('--KD_c', type=float, default=0.0, help='Balancing weight between losses in KD.')
parser.add_argument('--KD_r', type=float, default=0.0, help='Balancing weight between losses in KD.')
parser.add_argument('--KD_multi', type=float, default=0.0, help='Balancing weight between losses in KD.')
parser.add_argument('--threshold', type=float, default=0.7, help='entropy threshold for teacher logit to select confident samples.')
parser.add_argument('--teacher_path', type=str, default=None, help='Teacher model path for Knowledge Distillation.')
parser.add_argument('--cluster_KD_mode', type=str, default='euc', help='euc or cos distance.')
parser.add_argument('--teacher_arch', type=str, default='resnet50', help='Teacher model arch for Knowledge Distillation.')
parser.add_argument('--featureKD', action='store_true',help='Whether to use feature KD')
parser.add_argument('--relationKD', action='store_true',help='Whether to use instance relation KD')
parser.add_argument('--clusterKD', action='store_true',help='Whether to use cluster KD')
parser.add_argument('--RKD', action='store_true',help='Whether to use RKD, cvpr19')
parser.add_argument('--FSP', action='store_true',help='Whether to use FSP')
parser.add_argument('--overhaul', action='store_true',help='Whether to use overhaul')
parser.add_argument('--multi', action='store_true',help='Whether to use multi position FKD')

# Adabn
parser.add_argument('--adabn', action='store_true', help='adabn')

#-- save ckpt at certain epochs
parser.add_argument('--save_epoch_list', nargs='+', help='Set save ckpt epoch', required=False)

#-- multi step lr mileston
parser.add_argument('--milestones', nargs='+', help='Set save ckpt epoch', required=False)

# --- optimizer: adabelief
parser.add_argument('--optimizer_custom', type=str, default=None, help='optimizer')

# S5.1
parser.add_argument('--teacher_not_finetuned', action='store_true', help='S5 teacher not finetune, need to replace fc')
parser.add_argument('--fc_classes', type=int, default=None, help='fc classes for student; if none, abandom original fc on D1')


# S5.2
parser.add_argument('--resume_from_fc_of_target_data', action='store_true', help='S5.2 resume from ckpt with fc of D2')
parser.add_argument('--resume_from_diff_shape_fc', action='store_true', help='S5.2 resume from ckpt with fc of different shape with 1000 nor target data')

# pytorch pretrained, for resnet34 on cub dataset
parser.add_argument('--pytorch_pretrained', action='store_true', help='pytorch_pretrained')

def main(args, store):
    '''Given arguments and a cox store, trains as a model. Check out the 
    argparse object in this file for argument options.
    '''
    import time

    starttime = time.asctime(time.localtime(time.time()))
    print("start: ", starttime)
    ds, train_loader, validation_loader = get_dataset_and_loaders(args)

    if args.per_class_accuracy:
        assert args.dataset in ['pets', 'caltech101', 'caltech256', 'flowers', 'aircraft'], \
            f'Per-class accuracy not supported for the {args.dataset} dataset.'

        # VERY IMPORTANT
        # We report the per-class accuracy using the validation
        # set distribution. So ignore the training accuracy (as you will see it go
        # beyond 100. Don't freak out, it doesn't really capture anything),
        # just look at the validation accuarcy
        args.custom_accuracy = get_per_class_accuracy(args, validation_loader)

    model, checkpoint = get_model(args, ds, fc_classes=args.fc_classes)

    if args.eval_only:
        return train.eval_model(args, model, validation_loader, store=store)

    update_params = freeze_model(model, freeze_level=args.freeze_level)

    print(f"Dataset: {args.dataset} | Model: {args.arch}")

    if args.teacher_path:
        import copy
        args_t = copy.deepcopy(args)
        args_t.arch = args.teacher_arch
        if not args.teacher_not_finetuned:
            teacher_model, _ = resume_finetuning_from_checkpoint(args_t, ds, args.teacher_path)
        else:
            # teacher_model, _ = model_utils.make_and_restore_model(
            #     arch=pytorch_models[args_t.arch](
            #         args_t.pytorch_pretrained) if args_t.arch in pytorch_models.keys() else args_t.arch,
            #     dataset=datasets.ImageNet(''), resume_path=args.teacher_path, pytorch_pretrained=args_t.pytorch_pretrained,
            #     add_custom_forward=args_t.arch in pytorch_models.keys())

            teacher_model, _ = model_utils.make_and_restore_model(
                arch=pytorch_models[args_t.arch](
                    args_t.pytorch_pretrained) if args_t.arch in pytorch_models.keys() else args_t.arch,
                dataset=datasets.ImageNet(''),
                pytorch_pretrained=args_t.pytorch_pretrained,
                add_custom_forward=args_t.arch in pytorch_models.keys())

            while hasattr(teacher_model, 'model'):
                teacher_model = teacher_model.model

            if args.fc_classes is None:
                # ---------- for S5.1 abandoming original fc on D1, and new fc with shape to D2
                print(f'[Replacing the last layer with {args_t.additional_hidden} '
                      f'hidden layers and 1 classification layer that fits the {args_t.dataset} dataset.]')
                teacher_model = fine_tunify.ft(
                    args_t.arch, teacher_model, ds.num_classes, args_t.additional_hidden)
                teacher_model, _ = model_utils.make_and_restore_model(arch=teacher_model, dataset=ds,
                                                                       add_custom_forward=args_t.additional_hidden > 0 or args_t.arch in pytorch_models.keys())
            else:
                # ---------- for S5.1 keep original fc on D1
                print('keep the last layer')
                teacher_model = fine_tunify.ft(
                    args_t.arch, teacher_model, args.fc_classes, args_t.additional_hidden) # fc_classes for D1 (Imagenet)
                teacher_model, _ = model_utils.make_and_restore_model(arch=teacher_model, dataset=ds, resume_path=args.teacher_path,
                                                                      add_custom_forward=args_t.additional_hidden > 0 or args_t.arch in pytorch_models.keys())

        print(f"Teacher Model: {args.teacher_arch}")
        # train_distill.eval_model(args, teacher_model, validation_loader, store=store)
        train_distill.train_model(args, model, (train_loader, validation_loader), store=store,
                          checkpoint=checkpoint, update_params=update_params, teacher_model=teacher_model)
    else:
        if args.adabn:
            train_distill.adabn_model(args, model, (train_loader, validation_loader), store=store,
                                      checkpoint=checkpoint, update_params=update_params)
        else:
            train_distill.train_model(args, model, (train_loader, validation_loader), store=store,
                      checkpoint=checkpoint, update_params=update_params)

    endtime = time.asctime(time.localtime(time.time()))
    print("start: ", starttime)
    print("end: ", endtime)



def get_per_class_accuracy(args, loader):
    '''Returns the custom per_class_accuracy function. When using this custom function         
    look at only the validation accuracy. Ignore trainig set accuracy.
    '''
    def _get_class_weights(args, loader):
        '''Returns the distribution of classes in a given dataset.
        '''
        if args.dataset in ['pets', 'flowers']:
            targets = loader.dataset.targets

        elif args.dataset in ['caltech101', 'caltech256']:
            targets = np.array([loader.dataset.ds.dataset.y[idx]
                                for idx in loader.dataset.ds.indices])

        elif args.dataset == 'aircraft':
            targets = [s[1] for s in loader.dataset.samples]

        counts = np.unique(targets, return_counts=True)[1]
        class_weights = counts.sum()/(counts*len(counts))
        return ch.Tensor(class_weights)

    class_weights = _get_class_weights(args, loader)

    def custom_acc(logits, labels):
        '''Returns the top1 accuracy, weighted by the class distribution.
        This is important when evaluating an unbalanced dataset. 
        '''
        batch_size = labels.size(0)
        maxk = min(5, logits.shape[-1])
        prec1, _ = helpers.accuracy(
            logits, labels, topk=(1, maxk), exact=True)

        normal_prec1 = prec1.sum(0, keepdim=True).mul_(100/batch_size)
        weighted_prec1 = prec1 * class_weights[labels.cpu()].cuda()
        weighted_prec1 = weighted_prec1.sum(
            0, keepdim=True).mul_(100/batch_size)

        return weighted_prec1.item(), normal_prec1.item()

    return custom_acc


def get_dataset_and_loaders(args):
    '''Given arguments, returns a datasets object and the train and validation loaders.
    '''
    if args.dataset in ['imagenet', 'stylized_imagenet']:
        ds = datasets.ImageNet(args.data)
        train_loader, validation_loader = ds.make_loaders(
            only_val=args.eval_only, batch_size=args.batch_size, workers=8)
    elif args.dataset == 'mix4':
        ds, train_loader, validation_loader = transfer_datasets.make_loaders_mix4(args.batch_size, 8)
        if type(ds) == int:
            new_ds = datasets.CIFAR("/tmp")
            new_ds.num_classes = ds
            new_ds.mean = ch.tensor([0., 0., 0.])
            new_ds.std = ch.tensor([1., 1., 1.])
            ds = new_ds
    elif args.cifar10_cifar10:
        ds = datasets.CIFAR('/tmp')
        train_loader, validation_loader = ds.make_loaders(
            only_val=args.eval_only, batch_size=args.batch_size, workers=8)
    else:
        ds, (train_loader, validation_loader) = transfer_datasets.make_loaders(
            args.dataset, args.batch_size, 8, args.subset)
        if type(ds) == int:
            new_ds = datasets.CIFAR("/tmp")
            new_ds.num_classes = ds
            new_ds.mean = ch.tensor([0., 0., 0.])
            new_ds.std = ch.tensor([1., 1., 1.])
            ds = new_ds
    return ds, train_loader, validation_loader


def resume_finetuning_from_checkpoint(args, ds, finetuned_model_path):
    '''Given arguments, dataset object and a finetuned model_path, returns a model
    with loaded weights and returns the checkpoint necessary for resuming training.
    '''
    print('[Resuming finetuning from a checkpoint...]')
    if args.dataset in list(transfer_datasets.DS_TO_FUNC.keys()) and not args.cifar10_cifar10:
        model, _ = model_utils.make_and_restore_model(
            arch=pytorch_models[args.arch](
                args.pytorch_pretrained) if args.arch in pytorch_models.keys() else args.arch,
            dataset=datasets.ImageNet(''), add_custom_forward=args.arch in pytorch_models.keys())
        while hasattr(model, 'model'):
            model = model.model
        model = fine_tunify.ft(
            args.arch, model, ds.num_classes, args.additional_hidden)
        model, checkpoint = model_utils.make_and_restore_model(arch=model, dataset=ds, resume_path=finetuned_model_path,
                                                               add_custom_forward=args.additional_hidden > 0 or args.arch in pytorch_models.keys())
    else:
        model, checkpoint = model_utils.make_and_restore_model(
            arch=args.arch, dataset=ds, resume_path=finetuned_model_path)
    return model, checkpoint


def get_model(args, ds, fc_classes=None):
    '''Given arguments and a dataset object, returns an ImageNet model (with appropriate last layer changes to 
    fit the target dataset) and a checkpoint.The checkpoint is set to None if noe resuming training.
    '''
    finetuned_model_path = os.path.join(
        args.out_dir, args.exp_name, 'checkpoint.pt.latest')
    if args.resume and os.path.isfile(finetuned_model_path):
        model, checkpoint = resume_finetuning_from_checkpoint(
            args, ds, finetuned_model_path)
    elif args.weight and os.path.isfile(args.weight_path):
        model, checkpoint = resume_finetuning_from_checkpoint(
            args, ds, args.weight_path)
        checkpoint = None
    else:
        if args.dataset in list(transfer_datasets.DS_TO_FUNC.keys()) and not args.cifar10_cifar10 \
                and not args.resume_from_fc_of_target_data and not args.resume_from_diff_shape_fc:
            # import ipdb
            # ipdb.set_trace(context=20)
            model, _ = model_utils.make_and_restore_model(
                arch=pytorch_models[args.arch](
                    args.pytorch_pretrained) if args.arch in pytorch_models.keys() else args.arch,
                dataset=datasets.ImageNet(''), resume_path=args.model_path, pytorch_pretrained=args.pytorch_pretrained,
                add_custom_forward=args.arch in pytorch_models.keys())
            checkpoint = None
        elif args.resume_from_fc_of_target_data:
            model, _ = resume_finetuning_from_checkpoint(args,ds,args.model_path)
            checkpoint = None
        elif args.resume_from_diff_shape_fc:
            model, _ = model_utils.make_and_restore_model(
                arch=pytorch_models[args.arch](
                    args.pytorch_pretrained) if args.arch in pytorch_models.keys() else args.arch,
                dataset=datasets.ImageNet(''), pytorch_pretrained=args.pytorch_pretrained,
                add_custom_forward=args.arch in pytorch_models.keys())
            while hasattr(model, 'model'):
                model = model.model
            model = fine_tunify.ft(
                args.arch, model, fc_classes, args.additional_hidden)
            model, checkpoint = model_utils.make_and_restore_model(arch=model, dataset=ds, resume_path=args.model_path,
                                                                   add_custom_forward=args.additional_hidden > 0 or args.arch in pytorch_models.keys())
            while hasattr(model, 'model'):
                model = model.model
            model = fine_tunify.ft(
                args.arch, model, ds.num_classes, args.additional_hidden)
            model, checkpoint = model_utils.make_and_restore_model(arch=model, dataset=ds,
                                                                   add_custom_forward=args.additional_hidden > 0 or args.arch in pytorch_models.keys())
            return model, None
        else:
            model, _ = model_utils.make_and_restore_model(arch=args.arch, dataset=ds,
                                                          resume_path=args.model_path, pytorch_pretrained=args.pytorch_pretrained)
            checkpoint = None

        if not args.no_replace_last_layer and not args.eval_only:
            # print(f'[Replacing the last layer with {args.additional_hidden} '
            #       f'hidden layers and 1 classification layer that fits the {args.dataset} dataset.]')
            while hasattr(model, 'model'):
                model = model.model
            if fc_classes is None:
                print(f'[Replacing the last layer with {args.additional_hidden} '
                      f'hidden layers and 1 classification layer that fits the {args.dataset} dataset.]')
                model = fine_tunify.ft(
                    args.arch, model, ds.num_classes, args.additional_hidden)
            else:
                print(f'[Replacing the last layer with {args.additional_hidden} '
                      f'hidden layers and 1 classification layer that fc has dimension of {fc_classes}.]')
                model = fine_tunify.ft(
                    args.arch, model, fc_classes, args.additional_hidden)
            model, checkpoint = model_utils.make_and_restore_model(arch=model, dataset=ds,
                                                                   add_custom_forward=args.additional_hidden > 0 or args.arch in pytorch_models.keys())
        else:
            print('[NOT replacing the last layer]')
    return model, checkpoint


def freeze_model(model, freeze_level):
    '''
    Freezes up to args.freeze_level layers of the model (assumes a resnet model)
    '''
    # Freeze layers according to args.freeze-level
    update_params = None
    if freeze_level != -1:
        # assumes a resnet architecture
        assert len([name for name, _ in list(model.named_parameters())
                    if f"layer{freeze_level}" in name]), "unknown freeze level (only {1,2,3,4} for ResNets)"
        update_params = []
        freeze = True
        for name, param in model.named_parameters():
            print(name, param.size())

            # if not freeze and f'layer{freeze_level}' not in name:
            if not freeze and f'layer{freeze_level}' not in name and 'emb' not in name and 'l2norm' not in name:
                print(f"[Appending the params of {name} to the update list]")
                update_params.append(param)
            else:
                param.requires_grad = False

            if freeze and f'layer{freeze_level}' in name:
                # if the freeze level is detected stop freezing onwards
                freeze = False
    return update_params


def args_preprocess(args):
    '''
    Fill the args object with reasonable defaults, and also perform a sanity check to make sure no
    args are missing.
    '''
    if args.adv_train and eval(args.eps) == 0:
        print('[Switching to standard training since eps = 0]')
        args.adv_train = 0

    if args.pytorch_pretrained:
        assert not args.model_path, 'You can either specify pytorch_pretrained or model_path, not together.'

    # CIFAR10 to CIFAR10 assertions
    if args.cifar10_cifar10:
        assert args.dataset == 'cifar10'

    if args.data != '':
        cs.CALTECH101_PATH = cs.CALTECH256_PATH = cs.PETS_PATH = cs.CARS_PATH = args.data
        cs.FGVC_PATH = cs.FLOWERS_PATH = cs.DTD_PATH = cs.SUN_PATH = cs.FOOD_PATH = cs.BIRDS_PATH = args.data

    ALL_DS = list(transfer_datasets.DS_TO_FUNC.keys()) + \
        ['imagenet', 'breeds_living_9', 'stylized_imagenet', 'mix4']
    assert args.dataset in ALL_DS

    # Important for automatic job retries on the cluster in case of premptions. Avoid uuids.
    assert args.exp_name != None

    # Preprocess args
    args = defaults.check_and_fill_args(args, defaults.CONFIG_ARGS, None)
    if not args.eval_only:
        args = defaults.check_and_fill_args(args, defaults.TRAINING_ARGS, None)
    if args.adv_train or args.adv_eval:
        args = defaults.check_and_fill_args(args, defaults.PGD_ARGS, None)
    args = defaults.check_and_fill_args(args, defaults.MODEL_LOADER_ARGS, None)

    return args


if __name__ == "__main__":
    args = parser.parse_args()
    args = args_preprocess(args)

    pytorch_models = {
        'alexnet': models.alexnet,
        'vgg16': models.vgg16,
        'vgg16_bn': models.vgg16_bn,
        'squeezenet': models.squeezenet1_0,
        'densenet': models.densenet161,
        'shufflenet': models.shufflenet_v2_x1_0,
        # 'mobilenet': models.mobilenet_v2,
        'mobilenet': mobilenet.mobilenetv2,
        'mobilenet_pre_relu': mobilenet.mobilenetv2_pre_relu,
        'mobilenet_pre_relu_multi': mobilenet.mobilenetv2_pre_relu_multi,
        'mnv2_regressor': mobilenet.mobilenetv2_regressor,
        'mnv2_pre_relu_regressor': mobilenet.mobilenetv2_pre_relu_regressor,
        'resnext50_32x4d': models.resnext50_32x4d,
        'mnasnet': models.mnasnet1_0,
        'resnet34': models.resnet34,
        'resnet18_feat': resnet.resnet18_feat,
        'resnet18_custom': resnet.resnet18,
        'resnet18_feat_pre_relu': resnet.resnet18_feat_pre_relu,
        'resnet50_feat': resnet.resnet50,
        'resnet50_feat_svd': resnet.resnet50_feat_svd_pre_relu,
        'resnet50_feat_pre_relu': resnet.resnet50_feat_pre_relu,
    }

    # Create store and log the args
    store = cox.store.Store(args.out_dir, args.exp_name)
    if 'metadata' not in store.keys:
        args_dict = args.__dict__
        schema = cox.store.schema_from_dict(args_dict)
        store.add_table('metadata', schema)
        store['metadata'].append_row(args_dict)
    else:
        print('[Found existing metadata in store. Skipping this part.]')
    main(args, store)
