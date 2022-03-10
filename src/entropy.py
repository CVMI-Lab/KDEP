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

from utils import constants as cs
from utils import fine_tunify, transfer_datasets
from efficientnet_utils import (
    round_filters,
    round_repeats,
    drop_connect,
    get_same_padding_conv2d,
    get_model_params,
    efficientnet_params,
    load_pretrained_weights,
    Swish,
    MemoryEfficientSwish,
    calculate_output_image_size
)

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

# added for Knowledge Distillation
parser.add_argument('--KD_T', type=float, default=4, help='Temperature in Knowledge Distillation.')
parser.add_argument('--KD_a', type=float, default=0.9, help='Balancing weight between losses in KD.')
parser.add_argument('--teacher_path', type=str, default=None, help='Teacher model path for Knowledge Distillation.')
parser.add_argument('--teacher_arch', type=str, default='resnet50', help='Teacher model arch for Knowledge Distillation.')

# S5.1
parser.add_argument('--teacher_not_finetuned', action='store_true', help='S5 teacher not finetune, need to replace fc')
parser.add_argument('--fc_classes', type=int, default=None, help='fc classes for student; if none, abandom original fc on D1')

# S5.2
parser.add_argument('--resume_from_fc_of_target_data', action='store_true', help='S5.2 resume from ckpt with fc of D2')

# pytorch pretrained, for resnet34 on cub dataset
parser.add_argument('--pytorch_pretrained', action='store_true', help='pytorch_pretrained')

def main(args, store):
    '''Given arguments and a cox store, trains as a model. Check out the 
    argparse object in this file for argument options.
    '''
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

    mode = 'emb' #'relation'

    m = 100
    from resnet import resnet50,resnet18, resnet34, resnet152, resnet50_feat_pca_pre_relu, resnet18_feat, resnet18_feat_pre_relu, resnet50_feat_pre_relu_multi, resnet50_feat_interpolate
    from efficientnet import EfficientNet, EfficientNetB4, EfficientNetB7
    # --- Active Learning to select subset
    # resnet = resnet50(pretrained=True, deep_base=False, initpath='pretrained-models/transformed_resnet-50-l2-eps0.ckpt').cuda()
    # select_by_feature_dis(resnet, m)
    # select_by_entropy(teacher_model, m)

    if mode == 'relation':
        # --- Verify feature graph relation similarity
        res50 = resnet50(pretrained=True, initpath='pretrained-models/transformed_resnet-50-l2-eps0.ckpt').cuda()
        res34 = resnet34(pretrained=True, initpath='pretrained-models/resnet34-333f7ec4.pth').cuda()
        res18 = resnet18(pretrained=True, initpath='pretrained-models/resnet-18-l2-eps0.ckpt.clean').cuda()
        res18_distill = resnet18(pretrained=True, initpath='pretrained-models/resnet-18-100k-distilled-ex20.2.e2.ckpt.clean').cuda()
        res18_RKD = resnet18(pretrained=True, initpath='pretrained-models/res18-100k-RKD-ex50.2.t3.ckpt.clean').cuda()
        res18_100k_full_iters_380 = resnet18(pretrained=True, initpath='outdir/ex53.1.t1.1/380.ckpt.attacker.clean').cuda()
        res18_100k_full_iters_760 = resnet18(pretrained=True, initpath='outdir/ex53.1.t1.1/760.ckpt.attacker.clean').cuda()
        res18_100k_full_iters = resnet18(pretrained=True, initpath='outdir/ex53.1.t1.1/checkpoint.pt.latest.attacker.clean').cuda()
        # res18_1cycle = resnet18(pretrained=True, initpath='outdir/ex52.3/checkpoint.pt.latest.clean').cuda()

        # train_distill.eval_model(args, res18_100k_full_iters, validation_loader, store=store, warp_robust_code=False, all_feat=False)
        # model_list = [res18_100k_full_iters_380, res18_100k_full_iters_760, res18_100k_full_iters, res18_RKD, res18_distill, res18, res34, res50]
        model_list = [res18_100k_full_iters_380, res18_100k_full_iters_760, res18_100k_full_iters, res18_RKD, res18_distill, res50]
        # model_list = [res18_100k_full_iters_380, res18_100k_full_iters_760, res18_100k_full_iters, res50]

        # finetuned_model = resnet50(num_classes=10, pretrained=True, deep_base=False,
        #                            initpath='pretrained-models/transformed_resnet50_ex5.2_cifar10.ckpt').cuda()

        eval_feature_relation(model_list, train_loader)

    elif mode == 'emb':
        # ---- "When does label smoothing help? vis"
        # res18 = resnet18_feat_pre_relu(pretrained=True, initpath='pretrained-models/ex49.8.t2.pt.attacker.clean', num_classes=1000).cuda() # pretrained 18
        # res18 = resnet18(pretrained=True, initpath='pretrained-models/resnet-18-l2-eps0.ckpt.clean').cuda() # pretrained 18
        # res18 = resnet18(pretrained=True, initpath='pretrained-models/ex53.1.t1.ckpt.attacker.clean').cuda() # base teacher distilled
        # res18 = resnet18(pretrained=True, initpath='pretrained-models/ex53.1.p3.attacker.clean').cuda() # Meal v2 distilled
        # res18 = resnet18(pretrained=True, initpath='pretrained-models/ex31.3.w2.clean').cuda() # 100k pretrained SO*
        # res18 = resnet18(pretrained=True, initpath='outdir/ex52.3/checkpoint.pt.latest.attacker.clean').cuda() # 100k 1cycle bad
        # res18 = resnet18(pretrained=True, initpath='outdir/ex52.3.w1/checkpoint.pt.latest.attacker.clean').cuda() # 100k 1cycle normal
        # res18 = resnet18(pretrained=True, initpath='outdir/ex53.2/checkpoint.pt.latest.clean').cuda() # 100k SO long iters
        # res18 = resnet18(pretrained=True, initpath='outdir/ex50.2.test2/checkpoint.pt.latest.attacker.clean').cuda() # 100k S5 90 epochs
        # res18 = resnet18(pretrained=True, initpath='outdir/ex20.2.t2/checkpoint.pt.latest.clean').cuda() # 100k S5 150 epochs
        # res18 = resnet18(pretrained=True, initpath='outdir/ex54.1/checkpoint.pt.latest.attacker.clean').cuda() # 100k S5 370 epochs
        # res18 = resnet18(pretrained=True, initpath='outdir/ex54.2/checkpoint.pt.latest.attacker.clean').cuda() # 100k S5 600 epochs
        # res18 = resnet18(pretrained=True, initpath='outdir/ex58.3.t1/checkpoint.pt.latest.attacker.clean').cuda() # 100k S5 600 epochs
        # res18 = resnet18(pretrained=True, initpath='outdir/ex49.6.t1/checkpoint.pt.latest.attacker.clean').cuda() #  R18+regressor KL loss feat before regressor
        # res18 = resnet18_feat(pretrained=True, initpath='outdir/ex49.6.t1/checkpoint.pt.latest.attacker.emb.clean').cuda() #  R18+regressor KL loss feat after regressor
        model = resnet50_feat_pca_pre_relu(pretrained=True, initpath='pretrained-models/transformed_resnet-50-l2-eps0.ckpt').cuda()
        # res50 = resnet50_feat_pca_pre_relu(pretrained=True, initpath='pretrained-models/transformed_resnet-50-l2-eps0.ckpt').cuda()
        # res50 = resnet50_feat_pca_pre_relu(pretrained=True, initpath='pretrained-models/resnet50-miil-21k-pretrained-trans.pth.clean').cuda()
        # res50 = resnet50_feat_interpolate(pretrained=True, initpath='pretrained-models/transformed_resnet-50-l2-eps0.ckpt').cuda()
        # res50 = resnet50_feat_pca_pre_relu(pretrained=True, initpath='pretrained-models/MEALV2_ResNet50_224_trans.pth.clean').cuda()
        # model = resnet50_feat_pca_pre_relu(pretrained=True, initpath='pretrained-models/transformed_resnet-50-l2-eps0.ckpt').cuda()
        # res50 = resnet50_feat_pca_pre_relu(pretrained=True, initpath='pretrained-models/resnet-50-l2-eps0.ckpt.custom.cub-adabn.attacker.clean').cuda()

        from models_rw.efficientnet import tf_efficientnetv2_m_in21ft1k, tf_efficientnetv2_b3, tf_efficientnetv2_b0, \
            tf_efficientnetv2_b2, \
            efficientnetv2_rw_t, tf_efficientnetv2_s, gc_efficientnetv2_rw_t, efficientnetv2_rw_s, efficientnetv2_rw_m, \
            tf_efficientnet_b3
        # effnet= tf_efficientnetv2_b3(pretrained=True).cuda() # todo: 79 val; 86 train good
        # effnet= tf_efficientnetv2_b2(pretrained=True).cuda()  # todo: 79 val; 8 train
        # effnet= tf_efficientnetv2_b0(pretrained=True).cuda()  # todo: 78 val; 80 train
        # model = efficientnetv2_rw_t(pretrained=True).cuda() # todo: 81 val; 85 train good
        # effnet= gc_efficientnetv2_rw_t(pretrained=True).cuda() # todo: 81 val; 85 train
        # effnet= efficientnetv2_rw_s(pretrained=True).cuda() # todo: 81 val; 87 train
        # effnet= efficientnetv2_rw_m(pretrained=True).cuda()
        # effnet= tf_efficientnet_b3(pretrained=True).cuda()

        # ResneSt
        from models_rw.resnest import resnest50d, resnest50d_1s4x24d, resnest50d_4s2x40d, resnest101e
        # model = resnest50d(pretrained=True).cuda()
        # model = resnest101e(pretrained=True).cuda()
        # model = resnest50d_4s2x40d(pretrained=True).cuda()
        # model = resnest50d_1s4x24d(pretrained=True).cuda()

        # NFNet
        from models_rw.nfnet import dm_nfnet_f0, eca_nfnet_l1, eca_nfnet_l0, eca_nfnet_l2, dm_nfnet_f1, nf_resnet50
        # model = dm_nfnet_f0(pretrained=True).cuda() # todo: 82.6 val; 87 train
        # model = eca_nfnet_l1(pretrained=True).cuda() # todo: 82.6 val; 89 train
        # model = eca_nfnet_l0(pretrained=True).cuda() # todo: 81.6 val; 86 train
        # model = dm_nfnet_f1(pretrained=True).cuda() # todo: 83.4 val; 87 train
        # model = nf_resnet50(pretrained=True).cuda() # todo: 83.4 val; 87 train

        # Swin
        from models_rw.swin_transformer import swin_base_patch4_window7_224, swin_small_patch4_window7_224, \
            swin_large_patch4_window7_224, swin_base_patch4_window7_224_in22k,swin_large_patch4_window7_224_in22k
        # model = swin_base_patch4_window7_224(pretrained=True).cuda()
        # model = swin_small_patch4_window7_224(pretrained=True).cuda()
        # model = swin_large_patch4_window7_224(pretrained=True).cuda()
        # model = swin_base_patch4_window7_224_in22k(pretrained=True).cuda()
        # model = swin_large_patch4_window7_224_in22k(pretrained=True).cuda()

        # res152
        from resnet import resnet152, resnet152_feat_pca_pre_relu
        # model = resnet152_feat_pca_pre_relu(pretrained=True, initpath="pretrained-models/resnet152-b121ed2d.pth").cuda()

        # microsoftvision.models.resnet50
        # model = resnet50(pretrained=True, initpath='pretrained-models/MicrosoftVision-ResNet50.pth').cuda()
        model = resnet50_feat_pca_pre_relu(pretrained=True, initpath='pretrained-models/MicrosoftVision-ResNet50.pth').cuda()
        from resnet import resnet50_var_biggest
        # model = resnet50_var_biggest(pretrained=True,initpath='pretrained-models/MicrosoftVision-ResNet50.pth').cuda()

        # swsl, mealv2 res50
        # model = resnet50_feat_pca_pre_relu(pretrained=True, initpath="pretrained-models/transformed_resnet-50-l2-eps0.ckpt").cuda()
        # model = resnet50_feat_pca_pre_relu(pretrained=True, initpath="pretrained-models/resnet50-facebook-swsl-81.2.pth").cuda()
        # model = resnet50_feat_pca_pre_relu(pretrained=True, initpath="pretrained-models/MEALV2_ResNet50_224_trans.pth.clean").cuda()
        # model = resnet50(pretrained=True, initpath="pretrained-models/resnet50-miil-21k-1k-ft_ex42.1.e1.ckpt.clean").cuda()

        #here

        # res18 pre relu
        from resnet import resnet18_feat_pre_relu, resnet18_feat_pre_relu_custom_init, resnet18_feat_pre_relu_regressor # 11M paras
        # model =resnet18_feat_pre_relu(pretrained=False).cuda()

        #----- paper vis
        # model =resnet18_feat_pre_relu(pretrained=True, initpath="pretrained-models/ex95.1.pt.clean").cuda() # SVD+PTS 90 epochs 10% data
        # model =resnet18_feat_pre_relu(pretrained=True, initpath="pretrained-models/ex49.2.t1.o1.m2.pt.clean").cuda() # before conv; 1x1 conv pre relu 90 epochs 10% data
        # model =resnet18_feat_pre_relu_regressor(pretrained=True, initpath="pretrained-models/ex49.2.t1.o1.m2.pt.keep_1x1conv.clean").cuda() # after conv; 1x1 conv pre relu 90 epochs 10% data

        # model = resnet18_feat_pre_relu(pretrained=True,initpath="pretrained-models/ex103.6.1.pt.clean").cuda()  # ImageNet-1k SVD+PTS 90 epochs 10% data
        # model = resnet18_feat_pre_relu(pretrained=True, initpath='outdir/ex49.6.t1/checkpoint.pt.latest.attacker.clean').cuda() # before conv;
        # model = resnet18_feat(pretrained=True, initpath='outdir/ex49.6.t1/checkpoint.pt.latest.attacker.emb.clean').cuda() # after conv

        # logit / feat ex103.2.1
        from resnet import resnet18
        # model = resnet18(pretrained=True, initpath="pretrained-models/ex103.2.1.pt.clean").cuda()

        # model =resnet18_feat_pre_relu(pretrained=True, initpath="pretrained-models/R18_matching_var_const-1.5_init.clean").cuda()
        # model=resnet18_feat_pre_relu_custom_init(pretrained=False).cuda()
        # model = resnet18_feat_pre_relu(pretrained=True, initpath='pretrained-models/seg/ex104.10.t3.pt.seg').cuda()
        # model =resnet18_feat_pre_relu(pretrained=True, initpath='pretrained-models/resnet-18-l2-eps0.ckpt.clean').cuda()
        # model =resnet18_feat_pre_relu(pretrained=True, initpath='outdir/ex40.1/checkpoint.pt.latest.clean').cuda()

        # mocov2 RN50
        # model = resnet50_feat_pca_pre_relu(pretrained=True,initpath='pretrained-models/mocov2_RN50_vissl_200epochs.pt.clean').cuda()

        # torch.save(model.state_dict(), 'pretrained-models/scratch-res18-random_init_clean.pt')
        # import ipdb
        # ipdb.set_trace(context=20)


        # clip RN50
        # from clip import RN50
        # model = RN50(pretrained=True, initpath='pretrained-models/RN50.clean').cuda()
        from clip.model import clip_vitB_16
        # model = clip_vitB_16(pretrained=True, initpath='pretrained-models/ViT-B-16_trans.pt').cuda()

        # mobilenetv2
        from mobilenet import mobilenetv2, mobilenetv2_pre_relu, mobilenetv2_pre_relu_custom_init # 3.5M paras
        # model = mobilenetv2().cuda()
        # model = mobilenetv2_pre_relu(pretrained=False).cuda()
        # model = mobilenetv2_pre_relu_custom_init(pretrained=False).cuda()
        # model = mobilenetv2_pre_relu(pretrained=True, initpath="pretrained-models/mobilev2_matching_var_const-1_init.clean").cuda()

        # self-supervised pretrained
        from resnet import resnet152
        # model = resnet50_feat_pca_pre_relu(pretrained=True, initpath='pretrained-models/deepclusterv2_800ep_pretrain.pth.clean').cuda()

        # regnet
        from models_rw.regnet import regnety_120
        # model = regnety_120(pretrained=True).cuda()

        # wide-resnet
        # from models_rw.resnet import wide_resnet50_2_pre_relu
        # model = wide_resnet50_2_pre_relu(pretrained=True).cuda()

        pytorch_total_params = sum(p.numel() for p in model.parameters())
        print(pytorch_total_params)

        # model, _ = model_utils.make_and_restore_model(
        #     arch=resnet18_feat_pre_relu(pretrained=True, initpath='pretrained-models/ex49.12.t3.pt.clean'),
        #     dataset=datasets.ImageNet(''),
        #     add_custom_forward=True)
        #
        # model, checkpoint = model_utils.make_and_restore_model(arch=model, dataset=ds,
        #                                                        add_custom_forward=True)

        # res152 = resnet152(pretrained=True, initpath='pretrained-models/resnet152-b121ed2d.pth').cuda()
        # res50 = resnet50(pretrained=True, initpath='pretrained-models/MEALV2_ResNet50_224_trans.pth.clean').cuda()

        # train_distill.eval_model(args, model, validation_loader, store=store, warp_robust_code=False,all_feat=False)
        # train_distill.eval_model(args, model, train_loader, store=store, warp_robust_code=False,all_feat=False)

        # vis_emb(model, train_loader)
        cluster(model, train_loader)
        # save_emb(res50, train_loader)
        # cluster(res50, train_loader)

        # cluster_multi(res50, train_loader)
        # cluster(res18, train_loader)

    if args.teacher_path:
        import copy
        args_t = copy.deepcopy(args)
        args_t.arch = args.teacher_arch
        if not args.teacher_not_finetuned:
            teacher_model, _ = resume_finetuning_from_checkpoint(args_t, ds, args.teacher_path)
        else:
            teacher_model, _ = model_utils.make_and_restore_model(
                arch=pytorch_models[args_t.arch](
                    args_t.pytorch_pretrained) if args_t.arch in pytorch_models.keys() else args_t.arch,
                dataset=datasets.ImageNet(''), resume_path=args.teacher_path, pytorch_pretrained=args_t.pytorch_pretrained,
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
        train_distill.entropy_count(args, teacher_model, validation_loader, store=store)
        m=100

        from resnet import resnet50


        # resnet = resnet.cuda()

        # train_distill.train_model(args, model, (train_loader, validation_loader), store=store,
        #                   checkpoint=checkpoint, update_params=update_params, teacher_model=teacher_model)
    else:

        train.train_model(args, model, (train_loader, validation_loader), store=store,
                      checkpoint=checkpoint, update_params=update_params)

    # os.system('hdfs dfs -put outdir/* hdfs://haruna/home/byte_arnold_hl_vc/user/ruifeihe/exp_filter/')
    # print("putting outdir/* to hdfs://haruna/home/byte_arnold_hl_vc/user/ruifeihe/exp_filter/")

def vis_emb(res50, loader):
    if int(os.environ.get("NOTEBOOK_MODE", 0)) == 1:
        from tqdm import tqdm_notebook as tqdm
    else:
        from tqdm import tqdm as tqdm
    iterator = tqdm(enumerate(loader), total=len(loader))
    # dataset = 'CUB.10'
    dataset = 'imgnet.10'
    num = '_1000'

    out_target = []
    out_output = []

    def sort_index(inp, target):
        sorted_target, indices = torch.sort(target)
        sorted_inp = inp[indices]
        return sorted_inp, sorted_target

    for i, (inp, target) in iterator:
        inp = inp.cuda()
        target = target.cuda()



        # inp, target = sort_index(inp, target)
        _, feat = res50(inp) # feat
        # feat, _  = res50(inp) # logit
        # import ipdb
        # ipdb.set_trace(context=20)

        # --------------------------  adjust feature

        def FCT2(feat, T=4.0, n=3.0):
            feat = torch.sign(feat) * torch.pow(torch.abs(feat/T), n)
            return feat

        def FCT3(feat, T=4.0, more_compact=False):
            if more_compact:
                feat = torch.sign(feat) * torch.exp(-1 * torch.abs(feat / T))
            else:
                feat = torch.sign(feat) * torch.exp(torch.abs(feat / T))
            return feat

        import torch.nn.functional as F
        # feat = F.softmax(feat/1, dim=1)
        # feat = torch.sigmoid(feat)
        # feat = torch.exp(torch.exp(feat/1))

        # FCT2
        # feat = FCT2(feat, T=16, n=3) # more diverse
        # feat = FCT2(feat, T=0.1, n=1/2) # more compact
        #-- orig--:  feat = torch.sign(feat) * torch.pow(torch.abs(feat/1e-2),1/3)
        #-- orig--: feat = torch.sign(feat) * torch.pow(torch.abs(feat/16), 3)

        # FCT3
        # feat = FCT3(feat, T=16, more_compact=False) # more diverse

        # feat = torch.tanh(feat)


        # feat, _ = res50(inp) # logit
        output_array = feat.data.cpu().numpy()
        target_array = target.data.cpu().numpy()
        out_output.append(output_array)
        out_target.append(target_array[:, np.newaxis])


    output_array = np.concatenate(out_output, axis=0)
    # target_array = np.concatenate(out_target, axis=0)[:,0]
    target_array = np.concatenate(out_target, axis=0)

    # out_tensor = torch.from_numpy(output_array)  # B 2048
    # target_tensor = torch.from_numpy(target_array)
    #
    # out_tensor, target_tensor = sort_index(out_tensor, target_tensor)
    #
    # output_array = out_tensor.numpy()
    # target_array = target_tensor.numpy()

    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    # import ipdb
    # ipdb.set_trace(context=20)

    # ------ previous draw
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    output_array = tsne.fit_transform(output_array)
    plt.rcParams['figure.figsize'] = 10, 10
    plt.scatter(output_array[:, 0], output_array[:, 1], c=target_array[:,0])
    # ------ previous draw

    # ----- from https://towardsdatascience.com/visualising-high-dimensional-datasets-using-pca-and-t-sne-in-python-8ef87e7915b

    # import pandas as pd
    # import time
    # import seaborn as sns
    # from sklearn.decomposition import PCA
    # X = output_array
    # y = target_array
    # feat_cols = ['pixel' + str(i) for i in range(X.shape[1])]
    # df = pd.DataFrame(X, columns=feat_cols)
    # df['y'] = y
    # df['label'] = df['y'].apply(lambda i: str(i))
    # X, y = None, None
    # print('Size of the dataframe: {}'.format(df.shape))
    #
    #
    #
    # np.random.seed(42)
    # rndperm = np.random.permutation(df.shape[0])
    #
    # pca = PCA(n_components=3)
    # pca_result = pca.fit_transform(df[feat_cols].values)
    # df['pca-one'] = pca_result[:, 0]
    # df['pca-two'] = pca_result[:, 1]
    # df['pca-three'] = pca_result[:, 2]
    #
    # # plt.figure(figsize=(16, 10))
    # # sns.scatterplot(
    # #     x="pca-one", y="pca-two",
    # #     hue="y",
    # #     palette=sns.color_palette("hls", 7),
    # #     data=df.loc[rndperm, :],
    # #     legend="full",
    # #     alpha=0.3
    # # )
    # #
    # # title = 'new_sns-pca-cls-7_micro-R50-2048d-'
    # #
    # # title = title + dataset + num
    # # plt.title(title)
    # # plt.savefig('outdir/figs/' + title + '.png', bbox_inches='tight')
    #
    # # import ipdb
    # # ipdb.set_trace(context=20)
    #
    #
    # N = 1000
    # df_subset = df.loc[rndperm[:N], :].copy()
    #
    # data_subset = df_subset[feat_cols].values
    #
    # time_start = time.time()
    # tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=500)
    # tsne_results = tsne.fit_transform(data_subset)
    # print('t-SNE done! Time elapsed: {} seconds'.format(time.time() - time_start))
    #
    #
    #
    # df_subset['tsne-2d-one'] = tsne_results[:, 0]
    # df_subset['tsne-2d-two'] = tsne_results[:, 1]
    # plt.figure(figsize=(16, 10))
    # sns.scatterplot(
    #     x="tsne-2d-one", y="tsne-2d-two",
    #     hue="y",
    #     palette=sns.color_palette("hls", 10),
    #     data=df_subset,
    #     legend="full",
    #     alpha=1
    # )
    # ------- from https://towardsdatascience.com/visualising-high-dimensional-datasets-using-pca-and-t-sne-in-python-8ef87e7915b


    # ----- previous draw using from https://towardsdatascience.com/visualising-high-dimensional-datasets-using-pca-and-t-sne-in-python-8ef87e7915b

    # import pandas as pd
    # import time
    # import seaborn as sns
    # from sklearn.decomposition import PCA
    # X = output_array
    # y = target_array
    # feat_cols = ['pixel' + str(i) for i in range(X.shape[1])]
    # df = pd.DataFrame(X, columns=feat_cols)
    # df['y'] = y
    # df['label'] = df['y'].apply(lambda i: str(i))
    # X, y = None, None
    # print('Size of the dataframe: {}'.format(df.shape))
    #
    #
    #
    # np.random.seed(42)
    # rndperm = np.random.permutation(df.shape[0])
    #
    # pca = PCA(n_components=3)
    # pca_result = pca.fit_transform(df[feat_cols].values)
    # df['pca-one'] = pca_result[:, 0]
    # df['pca-two'] = pca_result[:, 1]
    # df['pca-three'] = pca_result[:, 2]
    #
    # N = 1000
    # df_subset = df.loc[rndperm[:N], :].copy()
    #
    # data_subset = df_subset[feat_cols].values
    #
    # time_start = time.time()
    #
    # tsne = TSNE(n_components=2, init='pca', random_state=0)
    # tsne_results = tsne.fit_transform(data_subset)
    # print('t-SNE done! Time elapsed: {} seconds'.format(time.time() - time_start))
    #
    #
    #
    # df_subset['tsne-2d-one'] = tsne_results[:, 0]
    # df_subset['tsne-2d-two'] = tsne_results[:, 1]
    # plt.figure(figsize=(10, 10))
    # sns.scatterplot(
    #     x="tsne-2d-one", y="tsne-2d-two",
    #     hue="y",
    #     palette=sns.color_palette("hls", 10),
    #     data=df_subset,
    #     legend="full",
    #     alpha=1
    # )
    # -----------------------------------------------------------


    # title = 'micro-res50-pre-relu-pca_512d_exp_T-48-0.9_feat'
    # title = 'base-res50-FCT2-n-(1div_3)-T-0.01-512d-'
    # title = 'cls-5_micro-R50-512d-FCT2-n-3-T-0.1'
    # title = 'cls-7_R18_before_1x1conv_overhaul-'
    # title = 'cls-7_R18_ex95.1-'
    # title = 'sns9100-tsne-cls-7_micro-R50-512d-svd-pts'
    # title = 'sns9100-tsne-cls-7_R18-ex95.1-'
    # title = 'sns9100-tsne-cls-7_R18-before-1x1-'
    # title = 'sns9100-tsne-cls-7_R18-after-1x1-'
    # title = 'sns9100-tsne-cls-7_Imagenet1k-R50-'
    # title = 'sns9100-tsne-cls-7_SWSL-R50-'
    # title = 'sns9100-tsne-cls-7_MEALv2-R50-'
    # title = 'sns9100-tsne-cls-7_swin-b-R50-'

    title = 'sns-tsne-micro-R50-2048d-svd-512d-pts'
    title = 'sns-tsne-R18-ex95.1-'
    title = 'sns-tsne-R18-before-1x1-'
    title = 'prev_draw_sns-tsne-R18-after-1x1-'
    title = 'prev_draw_sns-tsne-R18-before-1x1-'
    title = 'prev_draw_sns-tsne-R18-ex103.6.1-'
    title = 'prev_draw_swin-b-'
    title = 'prev_draw_logit-feat-'
    # title = 'prev_draw_logit-'


    #here

    title = title+dataset+num
    plt.title(title)
    plt.savefig('outdir/figs_10_1000/' + title + '.png', bbox_inches='tight')
    # plt.savefig('outdir/figs/logit_' + title + '.png', bbox_inches='tight')

    import ipdb
    ipdb.set_trace(context=20)

    a=0

def save_emb(res50, loader):
    if int(os.environ.get("NOTEBOOK_MODE", 0)) == 1:
        from tqdm import tqdm_notebook as tqdm
    else:
        from tqdm import tqdm as tqdm
    iterator = tqdm(enumerate(loader), total=len(loader))
    # dataset = 'CUB.10'
    dataset = 'imgnet.10'
    num = '_1000'

    out_target = []
    out_output = []
    for i, (inp, target) in iterator:
        inp = inp.cuda()
        target = target.cuda()

        def sort_index(inp, target):
            sorted_target, indices = torch.sort(target)
            sorted_inp = inp[indices]
            return sorted_inp, sorted_target

        # inp, target = sort_index(inp, target)
        _, feat = res50(inp) # feat
        # feat, _ = res50(inp) # logit
        output_array = feat.data.cpu().numpy()
        target_array = target.data.cpu().numpy()
        out_output.append(output_array)
        out_target.append(target_array[:, np.newaxis])

    output_array = np.concatenate(out_output, axis=0)
    target_array = np.concatenate(out_target, axis=0)[:,0]

    out_tensor = torch.from_numpy(output_array) # B 2048
    import ipdb
    ipdb.set_trace(context=20)

    mode = 'pca'
    # ---- pca save
    if mode == 'pca':
        feat_mean = torch.mean(out_tensor,dim=0)
        out_tensor = out_tensor - feat_mean
        u, s, vh = torch.svd(out_tensor)
        VT = vh.numpy()
        # U, sigma, VT = np.linalg.svd(output_array)
        np.save('pretrained-models/V_res50_100k_subset_from_train_loader_rm-mean_pre_relu.npy', VT)
        np.save('pretrained-models/V_res50_100k_subset_from_train_loader_rm-mean_pre_relu_feat_mean.npy', feat_mean.numpy())
        # np.save('pretrained-models/V_res50_100k_subset_from_train_loader.npy', VT)
        # --- for PCA:
        # --- feat=feat-feat_mean.cuda()
        # --- feat_pca = torch.matmul(feat, v[:, :512].cuda()) k=512

    elif mode == 'lda':
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        X = output_array
        Y = target_array
        lda = LinearDiscriminantAnalysis(n_components=512)
        lda.fit_transform(X, Y)

        print(lda.xbar_)
        print(lda.scalings_)

        np.save('pretrained-models/lda_res50_100k_subset_from_train_loader_xbar.npy', lda.xbar_) # 2048
        np.save('pretrained-models/lda_res50_100k_subset_from_train_loader_scalings.npy', lda.scalings_) # 2048 * 999
        # --- for lda:
        # --- feat_=feat-xbar.cuda()
        # --- feat_lda = torch.matmul(feat_, scalings.cuda())[:, :512]  k=512

    elif mode == 'pca_sklearn':
        feat_mean = torch.mean(out_tensor,dim=0)
        out_tensor = out_tensor - feat_mean
        from sklearn.decomposition import PCA
        pca = PCA(n_components=512,whiten=False)
        pca.fit(out_tensor)

        u, s, vh = torch.svd(out_tensor)
        VT = vh.numpy()
        # U, sigma, VT = np.linalg.svd(output_array)
        np.save('pretrained-models/V_res50_100k_subset_from_train_loader_rm-mean.npy', VT)
        np.save('pretrained-models/V_res50_100k_subset_from_train_loader_rm-mean_feat_mean.npy', feat_mean.numpy())
        # np.save('pretrained-models/V_res50_100k_subset_from_train_loader.npy', VT)
        # --- for PCA:
        # --- feat=feat-feat_mean.cuda()
        # --- feat_pca = torch.matmul(feat, vh[:, :512].cuda()) k=512

    elif mode == 'mag':
        # ----- choose by magnitude
        feat_mean = torch.mean(out_tensor, dim=0)
        sorted_feat, indices = torch.sort(feat_mean, descending=True)
        indices = torch.flip(indices,0)
        indices = indices[::-1]
        torch.save(indices, 'pretrained-models/V_res50_100k_subset_from_train_loader_magnitude_indices.pt')
        ret = torch.load('pretrained-models/V_res50_100k_subset_from_train_loader_magnitude_indices.pt')
        # --- for select:
        # --- feat_mag = torch.index_select(feat, 1, ret[:512].cuda(CUDA_LAUNCH_BLOCKING=1))

    elif mode == 'var':
        # ----- choose by magnitude
        # feat_mean = torch.mean(out_tensor, dim=0)
        feat_var = torch.var(out_tensor, dim=0)
        sorted_feat, indices = torch.sort(feat_var, descending=True)
        torch.save(indices, 'pretrained-models/V_res50_100k_subset_from_train_loader_var_indices.pt')
        ret = torch.load('pretrained-models/V_res50_100k_subset_from_train_loader_var_indices.pt')
        # --- for select:
        # --- feat_mag = torch.index_select(feat, 1, ret[:512].cuda(CUDA_LAUNCH_BLOCKING=1))



    a=0

def cluster(res50, loader):
    if int(os.environ.get("NOTEBOOK_MODE", 0)) == 1:
        from tqdm import tqdm_notebook as tqdm
    else:
        from tqdm import tqdm as tqdm
    iterator = tqdm(enumerate(loader), total=len(loader))
    # dataset = 'CUB.10'
    dataset = 'imgnet.10'
    num = '_1000'

    res50.eval()

    def sort_index(inp, target):
        sorted_target, indices = torch.sort(target)
        sorted_inp = inp[indices]
        return sorted_inp, sorted_target

    out_target = []
    out_output = []
    for i, (inp, target) in iterator:
        inp = inp.cuda()
        target = target.cuda()
        # import ipdb
        # ipdb.set_trace(context=20)
        # inp, target = sort_index(inp, target)
        _, feat = res50(inp) # feat

        # --------------------------  adjust feature

        def FCT2(feat, T=4.0, n=3.0):
            feat = torch.sign(feat) * torch.pow(torch.abs(feat / T), n)
            return feat

        def FCT3(feat, T=4.0, more_compact=False):
            if more_compact:
                feat = torch.sign(feat) * torch.exp(-1 * torch.abs(feat / T))
            else:
                feat = torch.sign(feat) * torch.exp(torch.abs(feat / T))
            return feat

        import torch.nn.functional as F
        # feat = F.softmax(feat/1, dim=1)
        # feat = torch.sigmoid(feat)
        # feat = torch.exp(torch.exp(feat/1))

        # FCT2
        # feat = FCT2(feat, T=16, n=3) # more diverse
        # feat = FCT2(feat, T=1e-1, n=1/3) # more compact
        # -- orig--:  feat = torch.sign(feat) * torch.pow(torch.abs(feat/1e-2),1/3)
        # -- orig--: feat = torch.sign(feat) * torch.pow(torch.abs(feat/16), 3)

        # FCT3
        # feat = FCT3(feat, T=16, more_compact=False)  # more diverse



        # output_teacher, _ = res50(inp, target=target, make_adv=False)

        # feat = output_teacher[0][1]
        # feat, _ = res50(inp) # logit
        output_array = feat.data.cpu().numpy()
        target_array = target.data.cpu().numpy()
        out_output.append(output_array)
        out_target.append(target_array[:, np.newaxis])

    output_array = np.concatenate(out_output, axis=0)
    target_array = np.concatenate(out_target, axis=0)[:,0]

    # import ipdb
    # ipdb.set_trace(context=20)
    area_gt_10 = (output_array < 0).sum() / 100000 / 512 * 100

    # np.save('pretrained-models/output_array_imgnet-100k-miil-21k.npy', output_array)
    # np.save('pretrained-models/target_array_imgnet-100k-miil-21k.npy', target_array)
    #
    # output_array = np.load('pretrained-models/output_array_imgnet-100k-miil-21k.npy')
    # # area_gt_10 = (output_array > 0).sum() / 100000 / 2048 * 100
    #
    # target_array = np.load('pretrained-models/target_array_imgnet-100k-miil-21k.npy')

    out_tensor = torch.from_numpy(output_array) # B 2048
    target_tensor = torch.from_numpy(target_array)

    out_tensor, target_tensor = sort_index(out_tensor, target_tensor)

    std = out_tensor.std(0)

    import ipdb
    ipdb.set_trace(context=20)

    # out_p = (out_tensor >0)
    # out_p_number = out_p.sum()
    # out_p_sum = (out_p * out_tensor).sum()
    # out_p_mean = out_p_sum / out_p_number
    #
    # std = out_tensor.std(0)
    # sorted_target, indices = torch.sort(std, descending=True)
    # torch.save(indices, "pretrained-models/imgnet-R50-2048d_pre_relu_var_descend_indice.pt")



    mode = 'pca'
    # ---- pca save
    if mode == 'pca':
        feat_mean = torch.mean(out_tensor,dim=0)
        out_tensor = out_tensor - feat_mean
        f, ff, vh = torch.svd(out_tensor)
        VT = vh.numpy()
        # U, sigma, VT = np.linalg.svd(output_array)
        np.save('pretrained-models/V-micro-pass-128k_from_train_loader_rm-mean_pre_relu_no_random.npy', VT)
        np.save('pretrained-models/V-micro-pass-128k_from_train_loader_rm-mean_pre_relu_feat_mean_no_random.npy', feat_mean.numpy())
        np.save('pretrained-models/V-micro-pass-128k_from_train_loader_rm-mean_pre_relu_eigen_values_no_random.npy', ff.numpy())

        a=0

        # # np.save('pretrained-models/V_res50_100k_subset_from_train_loader.npy', VT)
        # # --- for PCA:
        # # --- feat=feat-feat_mean.cuda()
        # # --- feat_pca = torch.matmul(out_tensor, vh[:, :512])  k=512
        #

        # vh = np.load('pretrained-models/V_res50_100k_subset_from_train_loader_rm-mean_pre_relu_no_random.npy')
        # feat_mean = np.load('pretrained-models/V_res50_100k_subset_from_train_loader_rm-mean_pre_relu_feat_mean_no_random.npy')
        # vh = torch.from_numpy(vh)
        # feat_mean = torch.from_numpy(feat_mean)
        #
        # out_tensor = out_tensor.cuda() - feat_mean.cuda()
        # feat_pca = torch.matmul(out_tensor, vh[:, :512].cuda())
        # torch.save(feat_pca.cpu(), 'pretrained-models/feat_pca.pt')

        # -- counting
        # area_gt_10 = (feat_pca > 60).sum() / 100000 / 512 *100
        # print('greater than 10: {} %'.format(area_gt_10))
        # area_gt_5 = (feat_pca > 5).sum() / 100000 / 512 * 100
        # print('greater than 5: {} %'.format(area_gt_5))
        # area_gt_1 = (feat_pca > 1).sum() / 100000 / 512 * 100
        # print('greater than 1: {} %'.format(area_gt_1))
        # area_lt_0 = (feat_pca < -100).sum() / 100000 / 512 * 100
        # print('less than 1: {} %'.format(area_lt_0))


        # cluster_mode = 'fps+kmeans'
        cluster_mode = 'supervised'

        if cluster_mode == 'supervised':
            # feat_pca = torch.matmul(out_tensor.cuda(), vh[:, :512].cuda())
            feat_pca = out_tensor
            cluster_num = 1000
            centers = torch.zeros(cluster_num, 512)

            for i in range(cluster_num):

                cluster_i_indice = (target_tensor==i)
                cluster_i_feat = feat_pca[cluster_i_indice]
                print(cluster_i_feat.shape)
                center_i = cluster_i_feat.mean(0)
                centers[i] = center_i

            torch.save(centers, 'pretrained-models/micro-res50_pre-relu_feat_no_random_gt-clusters_100k-subset_imgnet.pt')

            import ipdb
            ipdb.set_trace(context=20)

            dis3 = ch.nn.functional.mse_loss(centers[0], centers[2], reduction='mean')
            dis3 = ch.nn.functional.mse_loss(centers[0], feat_pca[2].cpu(), reduction='mean')

            x= feat_pca[90].cpu().view(1,-1)
            y= centers

            f,m,d = x.size(0), y.size(0), x.size(1)

            x = x.unsqueeze(1).expand(f, m, d)
            # x = x.unsqueeze(1).expand(x.size(0), y.size(0), x.size(1))
            y = y.unsqueeze(0).expand(f, m, d)

            dis = torch.pow(x - y, 2).sum(2)
        elif cluster_mode == 'Kmeans++':
            feat_pca = torch.load('pretrained-models/feat_pca.pt').numpy()
            from sklearn.cluster import kmeans_plusplus
            import ipdb
            ipdb.set_trace(context=20)
            centers, indices = kmeans_plusplus(feat_pca, n_clusters=1000, random_state=0)

            centers = torch.from_numpy(centers)
            torch.save(centers, 'pretrained-models/res50_pre-relu_feat_no_random_clusters-Kmeans++_1000.pt')
        elif cluster_mode == 'fps+kmeans':
            feat_pca = torch.load('pretrained-models/feat_pca.pt').numpy()
            from sklearn.cluster import KMeans
            from torch_cluster import fps
            # pip install torch-cluster -f https://data.pyg.org/whl/torch-1.7.1+cu10.2.html
            # feat_pca_tensor = torch.from_numpy(feat_pca)
            feat_pca_tensor = out_tensor
            cluster_num = 5000
            import time
            endtime = time.asctime(time.localtime(time.time()))
            print(endtime)
            example_num = feat_pca_tensor.shape[0]
            fps_index = fps(feat_pca_tensor, ratio=cluster_num/example_num, random_start=False)
            cluster_centers_init_fps = feat_pca_tensor[fps_index]
            endtime = time.asctime(time.localtime(time.time()))
            print(endtime)

            kmeans = KMeans(n_clusters=cluster_num, init=cluster_centers_init_fps.numpy(), random_state=0).fit(feat_pca_tensor.numpy())
            centers = kmeans.cluster_centers_
            labels_check = kmeans.labels_

            centers = torch.from_numpy(centers)
            torch.save(centers, 'pretrained-models/micro_res50_pre-relu_feat_pca_FCT2_no_random_clusters-fps+Kmeans_imgnet-100k_{}.pt'.format(cluster_num))
            endtime = time.asctime(time.localtime(time.time()))
            print(endtime)

            import ipdb
            ipdb.set_trace(context=20)

            a=0

    elif mode == 'nmf':

        # ----- sklearn
        # from sklearn.decomposition import non_negative_factorization
        # # W, H, n_iter = non_negative_factorization(output_array, n_components=512, init = 'random', random_state = 0)
        # # W, H, n_iter = non_negative_factorization(output_array, n_components=512, init = 'nndsvd', random_state = 0)
        # W, H, n_iter = non_negative_factorization(output_array, n_components=512, init = 'nndsvdar', verbose=10)
        import ipdb
        ipdb.set_trace(context=20)
        # ----- https://pypi.org/project/nmf-torch/
        from nmf import run_nmf
        H, W, err = run_nmf(output_array, n_components=512)
        H, W, err = run_nmf(out_tensor, n_components=512)

        # ----- https://github.com/yoyololicon/pytorch-NMF
        # from torchnmf.nmf import NMF
        # S= out_tensor
        # R=512
        # net = NMF(S.shape, rank=R).cuda()
        # net.fit(S.cuda(), tol=1e-5, max_iter=1000,verbose=True)
        # W, H = net.W.detach().cpu().numpy(), net.H.squeeze().detach().cpu().numpy()


        W=np.load('pretrained-models/NMF_torchnmf_after_relu_no_random.npy')
        W_pinv = np.linalg.pinv(W)


        # np.save('pretrained-models/NMF_sklearn_after_relu_no_random.npy', W)

        #-- eval
        re=W.dot(H.transpose())
        dis = (S-re.transpose())**2

        # np.save('pretrained-models/NMF_torchnmf_after_relu_no_random.npy', W)
        # np.save('pretrained-models/V_res50_100k_subset_from_train_loader_rm-mean_pre_relu_feat_mean_no_random_cub-adabn.npy', feat_mean.numpy())
        # np.save('pretrained-models/V_res50_100k_subset_from_train_loader.npy', VT)
        # --- for PCA:
        # --- feat=feat-feat_mean.cuda()
        # --- feat_pca = torch.matmul(feat, v[:, :512].cuda()) k=512



        feat_pca = torch.matmul(out_tensor.cuda(), vh[:, :512].cuda())
        centers = torch.zeros(1000, 512)
        for i in range(1000):

            cluster_i_feat = feat_pca[i*100:(i+1)*100, :]
            center_i = cluster_i_feat.mean(0)
            centers[i] = center_i

        torch.save(centers, 'pretrained-models/res50_pre-relu_feat_no_random_clusters_gt_cub-adabn.pt')
        import ipdb
        ipdb.set_trace(context=20)
        dis3 = ch.nn.functional.mse_loss(centers[0], centers[2], reduction='mean')
        dis3 = ch.nn.functional.mse_loss(centers[0], feat_pca[2].cpu(), reduction='mean')

        x= feat_pca[202].cpu().view(1,-1)
        y= centers

        l,m,d = x.size(0), y.size(0), x.size(1)

        x = x.unsqueeze(1).expand(l, m, d)
        # x = x.unsqueeze(1).expand(x.size(0), y.size(0), x.size(1))
        y = y.unsqueeze(0).expand(l, m, d)

        dis = torch.pow(x - y, 2).sum(2)

    elif mode == 'lda':
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        X = output_array
        Y = target_array
        lda = LinearDiscriminantAnalysis(n_components=512)
        lda.fit_transform(X, Y)

        print(lda.xbar_)
        print(lda.scalings_)

        np.save('pretrained-models/lda_res50_100k_subset_from_train_loader_xbar.npy', lda.xbar_) # 2048
        np.save('pretrained-models/lda_res50_100k_subset_from_train_loader_scalings.npy', lda.scalings_) # 2048 * 999
        # --- for lda:
        # --- feat_=feat-xbar.cuda()
        # --- feat_lda = torch.matmul(feat_, scalings.cuda())[:, :512]  k=512

    elif mode == 'pca_sklearn':
        feat_mean = torch.mean(out_tensor,dim=0)
        out_tensor = out_tensor - feat_mean
        from sklearn.decomposition import PCA
        pca = PCA(n_components=512,whiten=False)
        pca.fit(out_tensor)

        u, s, vh = torch.svd(out_tensor)
        VT = vh.numpy()
        # U, sigma, VT = np.linalg.svd(output_array)
        np.save('pretrained-models/V_res50_100k_subset_from_train_loader_rm-mean.npy', VT)
        np.save('pretrained-models/V_res50_100k_subset_from_train_loader_rm-mean_feat_mean.npy', feat_mean.numpy())
        # np.save('pretrained-models/V_res50_100k_subset_from_train_loader.npy', VT)
        # --- for PCA:
        # --- feat=feat-feat_mean.cuda()
        # --- feat_pca = torch.matmul(feat, vh[:, :512].cuda()) k=512

    elif mode == 'mag':
        # ----- choose by magnitude
        feat_mean = torch.mean(out_tensor, dim=0)
        sorted_feat, indices = torch.sort(feat_mean, descending=True)
        indices = torch.flip(indices,0)
        indices = indices[::-1]
        torch.save(indices, 'pretrained-models/V_res50_100k_subset_from_train_loader_magnitude_indices.pt')
        ret = torch.load('pretrained-models/V_res50_100k_subset_from_train_loader_magnitude_indices.pt')
        # --- for select:
        # --- feat_mag = torch.index_select(feat, 1, ret[:512].cuda(CUDA_LAUNCH_BLOCKING=1))

    elif mode == 'var':
        # ----- choose by magnitude
        # feat_mean = torch.mean(out_tensor, dim=0)
        feat_var = torch.var(out_tensor, dim=0)
        sorted_feat, indices = torch.sort(feat_var, descending=True)
        torch.save(indices, 'pretrained-models/V_res50_100k_subset_from_train_loader_var_indices.pt')
        ret = torch.load('pretrained-models/V_res50_100k_subset_from_train_loader_var_indices.pt')
        # --- for select:
        # --- feat_mag = torch.index_select(feat, 1, ret[:512].cuda(CUDA_LAUNCH_BLOCKING=1))



    a=0

def cluster_multi(res50, loader):
    if int(os.environ.get("NOTEBOOK_MODE", 0)) == 1:
        from tqdm import tqdm_notebook as tqdm
    else:
        from tqdm import tqdm as tqdm
    iterator = tqdm(enumerate(loader), total=len(loader))
    # dataset = 'CUB.10'
    dataset = 'imgnet.10'
    num = '_1000'

    def sort_index(inp, target):
        sorted_target, indices = torch.sort(target)
        sorted_inp = inp[indices]
        return sorted_inp, sorted_target

    out_target = []
    out_output = []
    for i, (inp, target) in iterator:
        inp = inp.cuda()
        target = target.cuda()

        # inp, target = sort_index(inp, target)
        _, feat_list = res50(inp) # feat
        import ipdb
        ipdb.set_trace(context=20)
        # output_teacher, _ = res50(inp, target=target, make_adv=False)

        # feat = output_teacher[0][1]
        # feat, _ = res50(inp) # logit



        output_array = feat.data.cpu().numpy()
        target_array = target.data.cpu().numpy()
        out_output.append(output_array)
        out_target.append(target_array[:, np.newaxis])

    output_array = np.concatenate(out_output, axis=0)
    target_array = np.concatenate(out_target, axis=0)[:,0]

    # import ipdb
    # ipdb.set_trace(context=20)
    area_gt_10 = (output_array < 0).sum() / 100000 / 512 * 100

    # np.save('pretrained-models/output_array_imgnet-100k-miil-21k.npy', output_array)
    # np.save('pretrained-models/target_array_imgnet-100k-miil-21k.npy', target_array)
    #
    # output_array = np.load('pretrained-models/output_array_imgnet-100k-miil-21k.npy')
    # # area_gt_10 = (output_array > 0).sum() / 100000 / 2048 * 100
    #
    # target_array = np.load('pretrained-models/target_array_imgnet-100k-miil-21k.npy')

    out_tensor = torch.from_numpy(output_array) # B 2048
    target_tensor = torch.from_numpy(target_array)

    out_tensor, target_tensor = sort_index(out_tensor, target_tensor)



    mode = 'pca'
    # ---- pca save
    if mode == 'pca':
        feat_mean = torch.mean(out_tensor,dim=0)
        out_tensor = out_tensor - feat_mean
        u, s, vh = torch.svd(out_tensor)
        VT = vh.numpy()
        # U, sigma, VT = np.linalg.svd(output_array)
        np.save('pretrained-models/V_res50-mealv2_100k_subset_from_train_loader_rm-mean_pre_relu_no_random.npy', VT)
        np.save('pretrained-models/V_res50-mealv2_100k_subset_from_train_loader_rm-mean_pre_relu_feat_mean_no_random.npy', feat_mean.numpy())
        import ipdb
        ipdb.set_trace(context=20)
        a=0

        # # np.save('pretrained-models/V_res50_100k_subset_from_train_loader.npy', VT)
        # # --- for PCA:
        # # --- feat=feat-feat_mean.cuda()
        # # --- feat_pca = torch.matmul(feat, v[:, :512].cuda()) k=512
        #

        # vh = np.load('pretrained-models/V_res50_100k_subset_from_train_loader_rm-mean_pre_relu_no_random.npy')
        # feat_mean = np.load('pretrained-models/V_res50_100k_subset_from_train_loader_rm-mean_pre_relu_feat_mean_no_random.npy')
        # vh = torch.from_numpy(vh)
        # feat_mean = torch.from_numpy(feat_mean)
        #
        # out_tensor = out_tensor.cuda() - feat_mean.cuda()
        # feat_pca = torch.matmul(out_tensor, vh[:, :512].cuda())
        # torch.save(feat_pca.cpu(), 'pretrained-models/feat_pca.pt')

        # -- counting
        # area_gt_10 = (feat_pca > 60).sum() / 100000 / 512 *100
        # print('greater than 10: {} %'.format(area_gt_10))
        # area_gt_5 = (feat_pca > 5).sum() / 100000 / 512 * 100
        # print('greater than 5: {} %'.format(area_gt_5))
        # area_gt_1 = (feat_pca > 1).sum() / 100000 / 512 * 100
        # print('greater than 1: {} %'.format(area_gt_1))
        # area_lt_0 = (feat_pca < -100).sum() / 100000 / 512 * 100
        # print('less than 1: {} %'.format(area_lt_0))


        cluster_mode = 'fps+kmeans'
        # cluster_mode = 'supervised'

        if cluster_mode == 'supervised':
            # feat_pca = torch.matmul(out_tensor.cuda(), vh[:, :512].cuda())
            feat_pca = out_tensor
            cluster_num = 1000
            centers = torch.zeros(cluster_num, 512)

            for i in range(cluster_num):

                cluster_i_indice = (target_tensor==i)
                cluster_i_feat = feat_pca[cluster_i_indice]
                print(cluster_i_feat.shape)
                center_i = cluster_i_feat.mean(0)
                centers[i] = center_i

            torch.save(centers, 'pretrained-models/res50_pre-relu_feat_no_random_clusters_all_samples_imgnet-100k.pt')

            import ipdb
            ipdb.set_trace(context=20)

            dis3 = ch.nn.functional.mse_loss(centers[0], centers[2], reduction='mean')
            dis3 = ch.nn.functional.mse_loss(centers[0], feat_pca[2].cpu(), reduction='mean')

            x= feat_pca[90].cpu().view(1,-1)
            y= centers

            f,m,d = x.size(0), y.size(0), x.size(1)

            x = x.unsqueeze(1).expand(f, m, d)
            # x = x.unsqueeze(1).expand(x.size(0), y.size(0), x.size(1))
            y = y.unsqueeze(0).expand(f, m, d)

            dis = torch.pow(x - y, 2).sum(2)
        elif cluster_mode == 'Kmeans++':
            feat_pca = torch.load('pretrained-models/feat_pca.pt').numpy()
            from sklearn.cluster import kmeans_plusplus
            import ipdb
            ipdb.set_trace(context=20)
            centers, indices = kmeans_plusplus(feat_pca, n_clusters=1000, random_state=0)

            centers = torch.from_numpy(centers)
            torch.save(centers, 'pretrained-models/res50_pre-relu_feat_no_random_clusters-Kmeans++_1000.pt')
        elif cluster_mode == 'fps+kmeans':
            feat_pca = torch.load('pretrained-models/feat_pca.pt').numpy()
            from sklearn.cluster import KMeans
            from torch_cluster import fps
            # feat_pca_tensor = torch.from_numpy(feat_pca)
            feat_pca_tensor = out_tensor
            cluster_num = 20000
            import time
            endtime = time.asctime(time.localtime(time.time()))
            print(endtime)
            example_num = feat_pca_tensor.shape[0]
            fps_index = fps(feat_pca_tensor, ratio=cluster_num/example_num, random_start=False)
            cluster_centers_init_fps = feat_pca_tensor[fps_index]
            endtime = time.asctime(time.localtime(time.time()))
            print(endtime)

            kmeans = KMeans(n_clusters=cluster_num, init=cluster_centers_init_fps.numpy(), random_state=0).fit(feat_pca_tensor.numpy())
            centers = kmeans.cluster_centers_
            labels_check = kmeans.labels_

            centers = torch.from_numpy(centers)
            torch.save(centers, 'pretrained-models/res50_pre-relu_feat_no_random_clusters-fps+Kmeans_imgnet-100k_{}.pt'.format(cluster_num))
            endtime = time.asctime(time.localtime(time.time()))
            print(endtime)

            import ipdb
            ipdb.set_trace(context=20)

            a=0

    elif mode == 'nmf':

        # ----- sklearn
        # from sklearn.decomposition import non_negative_factorization
        # # W, H, n_iter = non_negative_factorization(output_array, n_components=512, init = 'random', random_state = 0)
        # # W, H, n_iter = non_negative_factorization(output_array, n_components=512, init = 'nndsvd', random_state = 0)
        # W, H, n_iter = non_negative_factorization(output_array, n_components=512, init = 'nndsvdar', verbose=10)
        import ipdb
        ipdb.set_trace(context=20)
        # ----- https://pypi.org/project/nmf-torch/
        from nmf import run_nmf
        H, W, err = run_nmf(output_array, n_components=512)
        H, W, err = run_nmf(out_tensor, n_components=512)

        # ----- https://github.com/yoyololicon/pytorch-NMF
        # from torchnmf.nmf import NMF
        # S= out_tensor
        # R=512
        # net = NMF(S.shape, rank=R).cuda()
        # net.fit(S.cuda(), tol=1e-5, max_iter=1000,verbose=True)
        # W, H = net.W.detach().cpu().numpy(), net.H.squeeze().detach().cpu().numpy()


        W=np.load('pretrained-models/NMF_torchnmf_after_relu_no_random.npy')
        W_pinv = np.linalg.pinv(W)


        # np.save('pretrained-models/NMF_sklearn_after_relu_no_random.npy', W)

        #-- eval
        re=W.dot(H.transpose())
        dis = (S-re.transpose())**2

        # np.save('pretrained-models/NMF_torchnmf_after_relu_no_random.npy', W)
        # np.save('pretrained-models/V_res50_100k_subset_from_train_loader_rm-mean_pre_relu_feat_mean_no_random_cub-adabn.npy', feat_mean.numpy())
        # np.save('pretrained-models/V_res50_100k_subset_from_train_loader.npy', VT)
        # --- for PCA:
        # --- feat=feat-feat_mean.cuda()
        # --- feat_pca = torch.matmul(feat, v[:, :512].cuda()) k=512



        feat_pca = torch.matmul(out_tensor.cuda(), vh[:, :512].cuda())
        centers = torch.zeros(1000, 512)
        for i in range(1000):

            cluster_i_feat = feat_pca[i*100:(i+1)*100, :]
            center_i = cluster_i_feat.mean(0)
            centers[i] = center_i

        torch.save(centers, 'pretrained-models/res50_pre-relu_feat_no_random_clusters_gt_cub-adabn.pt')
        import ipdb
        ipdb.set_trace(context=20)
        dis3 = ch.nn.functional.mse_loss(centers[0], centers[2], reduction='mean')
        dis3 = ch.nn.functional.mse_loss(centers[0], feat_pca[2].cpu(), reduction='mean')

        x= feat_pca[202].cpu().view(1,-1)
        y= centers

        l,m,d = x.size(0), y.size(0), x.size(1)

        x = x.unsqueeze(1).expand(l, m, d)
        # x = x.unsqueeze(1).expand(x.size(0), y.size(0), x.size(1))
        y = y.unsqueeze(0).expand(l, m, d)

        dis = torch.pow(x - y, 2).sum(2)

    elif mode == 'lda':
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        X = output_array
        Y = target_array
        lda = LinearDiscriminantAnalysis(n_components=512)
        lda.fit_transform(X, Y)

        print(lda.xbar_)
        print(lda.scalings_)

        np.save('pretrained-models/lda_res50_100k_subset_from_train_loader_xbar.npy', lda.xbar_) # 2048
        np.save('pretrained-models/lda_res50_100k_subset_from_train_loader_scalings.npy', lda.scalings_) # 2048 * 999
        # --- for lda:
        # --- feat_=feat-xbar.cuda()
        # --- feat_lda = torch.matmul(feat_, scalings.cuda())[:, :512]  k=512

    elif mode == 'pca_sklearn':
        feat_mean = torch.mean(out_tensor,dim=0)
        out_tensor = out_tensor - feat_mean
        from sklearn.decomposition import PCA
        pca = PCA(n_components=512,whiten=False)
        pca.fit(out_tensor)

        u, s, vh = torch.svd(out_tensor)
        VT = vh.numpy()
        # U, sigma, VT = np.linalg.svd(output_array)
        np.save('pretrained-models/V_res50_100k_subset_from_train_loader_rm-mean.npy', VT)
        np.save('pretrained-models/V_res50_100k_subset_from_train_loader_rm-mean_feat_mean.npy', feat_mean.numpy())
        # np.save('pretrained-models/V_res50_100k_subset_from_train_loader.npy', VT)
        # --- for PCA:
        # --- feat=feat-feat_mean.cuda()
        # --- feat_pca = torch.matmul(feat, vh[:, :512].cuda()) k=512

    elif mode == 'mag':
        # ----- choose by magnitude
        feat_mean = torch.mean(out_tensor, dim=0)
        sorted_feat, indices = torch.sort(feat_mean, descending=True)
        indices = torch.flip(indices,0)
        indices = indices[::-1]
        torch.save(indices, 'pretrained-models/V_res50_100k_subset_from_train_loader_magnitude_indices.pt')
        ret = torch.load('pretrained-models/V_res50_100k_subset_from_train_loader_magnitude_indices.pt')
        # --- for select:
        # --- feat_mag = torch.index_select(feat, 1, ret[:512].cuda(CUDA_LAUNCH_BLOCKING=1))

    elif mode == 'var':
        # ----- choose by magnitude
        # feat_mean = torch.mean(out_tensor, dim=0)
        feat_var = torch.var(out_tensor, dim=0)
        sorted_feat, indices = torch.sort(feat_var, descending=True)
        torch.save(indices, 'pretrained-models/V_res50_100k_subset_from_train_loader_var_indices.pt')
        ret = torch.load('pretrained-models/V_res50_100k_subset_from_train_loader_var_indices.pt')
        # --- for select:
        # --- feat_mag = torch.index_select(feat, 1, ret[:512].cuda(CUDA_LAUNCH_BLOCKING=1))



    a=0


def eval_feature_relation(model_list, loader):
    if int(os.environ.get("NOTEBOOK_MODE", 0)) == 1:
        from tqdm import tqdm_notebook as tqdm
    else:
        from tqdm import tqdm as tqdm
    iterator = tqdm(enumerate(loader), total=len(loader))
    # dataset = 'CUB.10'
    dataset = 'imgnet'
    batch_size = 128
    for i, (inp, target) in iterator:
        inp = inp.cuda()
        target = target.cuda()

        def sort_index(inp, target):
            sorted_target, indices = torch.sort(target)
            sorted_inp = inp[indices]
            return sorted_inp, target

        inp, target = sort_index(inp, target)

        def sim_matrix(a, b, eps=1e-8):
            """
            added eps for numerical stability
            """
            a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
            a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
            b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
            sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
            return sim_mt

        def get_relation(feat):
            cos_sim = sim_matrix(feat, feat)
            return cos_sim

        def plot(R, path):
            import matplotlib.pyplot as plt
            plt.matshow(R.detach().cpu().numpy(), cmap=plt.get_cmap('Greys'), alpha=0.5)  # , alpha=0.3
            plt.title(path.split('/')[-1][:-4])
            plt.savefig(path)

        def get_relation_and_plot(model, name='res18'):
            # _, feat = model(inp)
            feat, _ = model(inp)
            T=4
            feat = ch.nn.functional.softmax(feat/T, dim=1)
            # import ipdb
            # ipdb.set_trace(context=20)
            R = get_relation(feat)
            plot(R,'outdir/feat_figures_distill/logit-{}-on-{}-bs-{}.png'.format(name, dataset, batch_size))
            return R

        # name_list = ['100k-full-iters-res18-380', '100k-full-iters-res18-760','100k-full-iters-res18','RKD-res18','Distilled-res18','Pretrained-res18','Pretrained-res34','Pretrained-res50']
        name_list = ['100k-full-iters-res18-380', '100k-full-iters-res18-760','100k-full-iters-res18','RKD-res18','Distilled-res18','Pretrained-res50']
        # name_list = ['100k-full-iters-res18-380', '100k-full-iters-res18-760','100k-full-iters-res18','Pretrained-res50']
        sim_dict = {}
        for i in range(len(name_list)):
            sim_dict[name_list[i]] = get_relation_and_plot(model_list[i], name_list[i])
        print('\n')
        dis00 = ch.nn.functional.mse_loss(sim_dict['100k-full-iters-res18'], sim_dict['Pretrained-res50'], reduction='mean')
        dis01 = ch.nn.functional.mse_loss(sim_dict['100k-full-iters-res18-380'], sim_dict['Pretrained-res50'], reduction='mean')
        dis02 = ch.nn.functional.mse_loss(sim_dict['100k-full-iters-res18-760'], sim_dict['Pretrained-res50'], reduction='mean')
        # dis00 = ch.nn.functional.mse_loss(sim_dict['Distilled-res18'], sim_dict['RKD-res18'], reduction='mean')
        print(dis00)
        print(dis01)
        print(dis02)
        dis0 = ch.nn.functional.mse_loss(sim_dict['RKD-res18'], sim_dict['Pretrained-res50'], reduction='mean')

        # dis1 = ch.nn.functional.mse_loss(sim_dict['Distilled-res18'], sim_dict['Pretrained-res18'], reduction='mean')
        # dis2 = ch.nn.functional.mse_loss(sim_dict['Distilled-res18'], sim_dict['Pretrained-res34'], reduction='mean')
        dis3 = ch.nn.functional.mse_loss(sim_dict['Distilled-res18'], sim_dict['Pretrained-res50'], reduction='mean')
        # dis4 = ch.nn.functional.mse_loss(sim_dict['Pretrained-res18'], sim_dict['Pretrained-res50'], reduction='mean')
        # dis5 = ch.nn.functional.mse_loss(sim_dict['Pretrained-res34'], sim_dict['Pretrained-res50'], reduction='mean')
        # dis6 = ch.nn.functional.mse_loss(sim_dict['Pretrained-res34'], sim_dict['Pretrained-res18'], reduction='mean')

        print(dis0)
        # print(dis1)
        # print(dis2)
        print(dis3)
        # print(dis4)
        # print(dis5)
        # print(dis6)

        import ipdb
        ipdb.set_trace(context=20)
        a=0

def select_by_entropy(teacher_model, m):
    data_root = 'data/imagenet/all_images'
    import glob
    import os
    import cv2
    from torchvision import transforms
    from PIL import Image

    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]

    TRAIN_TRANSFORMS = transforms.Compose([
        # transforms.Resize(32),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
        # transforms.Normalize(mean=mean, std=std),
    ])

    # os.system('mv n*tar ILSVRC2012_img_train/')
    categories = sorted(glob.glob(data_root+'/n*'))
    number = 1000
    workers = 8
    number_each = number // workers
    worker_id = 6
    # import ipdb
    # ipdb.set_trace(context=20)
    # categories = categories[number_each*worker_id:number_each*(worker_id+1)]


    m_shot_path_name = 'AL_100k_cf_low'
    os.system('mkdir {}/../{}'.format(data_root,m_shot_path_name))
    batch_size = 64


    for category in categories:
        entropy_dict = {}

        print("selecting category %s"%category)
        category_name = category.split('/')[-1]
        os.system('mkdir {}/../{}/{}'.format(data_root, m_shot_path_name, category_name))
        images_paths = sorted(glob.glob(category+'/*'))
        cnt=0
        times = -1
        img_batch = torch.zeros(batch_size, 3, 224,224).cuda()
        for img_path in images_paths:
            # img = cv2.imread(img_path)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # img = Image.fromarray(np.uint8(img))

            img = Image.open(img_path)
            img = TRAIN_TRANSFORMS(img)
            img = img.cuda()
            # img = torch.unsqueeze(img,dim=0)
            if img.shape[0] ==4:
                # import ipdb
                # ipdb.set_trace(context=20)
                img = img[:3]
            img_batch[cnt] = img
            cnt+=1

            del img
            torch.cuda.empty_cache()


            if cnt == batch_size:

                times+=1
                cnt=0
                model_logits, _ = teacher_model(img_batch)
                Temperature = 1
                model_logits = model_logits / Temperature

                model_confidence = ch.nn.functional.softmax(model_logits, dim=1)  # B x 1000

                def get_entropy(confidence):
                    return -(confidence * ch.log(confidence)).sum(1)


                # entropy = get_entropy(model_confidence)
                entropy = torch.max(model_confidence, dim=1)[0]

                for i in range(batch_size):
                    img_path_name = images_paths[times*batch_size+i]
                    entropy_dict[img_path_name] = entropy[i].item()

                del img_batch, model_confidence, model_logits,entropy
                torch.cuda.empty_cache()
                img_batch = torch.zeros(batch_size, 3, 224, 224).cuda()

        # if not divided by batch_size
        if cnt!=0:
            # import ipdb
            # ipdb.set_trace(context=20)
            img_batch = img_batch[:cnt]

            model_logits, _ = teacher_model(img_batch)
            Temperature = 1
            model_logits = model_logits / Temperature

            model_confidence = ch.nn.functional.softmax(model_logits, dim=1)  # B x 1000

            def get_entropy(confidence):
                return -(confidence * ch.log(confidence)).sum(1)

            # entropy = get_entropy(model_confidence)
            entropy = torch.max(model_confidence,dim=1)[0]
            # import ipdb
            # ipdb.set_trace(context=20)

            for i in range(cnt):
                img_path_name = images_paths[times * batch_size + i]
                entropy_dict[img_path_name] = entropy[i].item()

            del img_batch, model_confidence, model_logits, entropy
            torch.cuda.empty_cache()

        # import ipdb
        # ipdb.set_trace(context=20)
        # sorted_list = sorted(entropy_dict.items(), key=lambda x: x[1], reverse=True)
        sorted_list = sorted(entropy_dict.items(), key=lambda x: x[1], reverse=False) # cf low
        selected_imgs = sorted_list[:m]

        for selected in selected_imgs:
            selected_path = selected[0]
            os.system('cp {} {}/../{}/{}/'.format(selected_path, data_root, m_shot_path_name, category_name))

def select_by_feature_dis(teacher_model, m):
    data_root = 'data/imagenet/all_images'
    import glob
    import os
    import cv2
    from torchvision import transforms
    from PIL import Image

    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]

    TRAIN_TRANSFORMS = transforms.Compose([
        # transforms.Resize(32),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
        # transforms.Normalize(mean=mean, std=std),
    ])

    # os.system('mv n*tar ILSVRC2012_img_train/')
    categories = sorted(glob.glob(data_root+'/n*'))
    number = 1000
    workers = 8
    number_each = number // workers
    worker_id = 6
    # import ipdb
    # ipdb.set_trace(context=20)
    # categories = categories[number_each*worker_id:number_each*(worker_id+1)]


    m_shot_path_name = 'AL_100k_feat'
    os.system('mkdir {}/../{}'.format(data_root,m_shot_path_name))
    batch_size = 64


    for category in categories:
        entropy_dict = {}

        print("selecting category %s"%category)
        import time
        starttime = time.asctime(time.localtime(time.time()))
        category_name = category.split('/')[-1]
        os.system('mkdir {}/../{}/{}'.format(data_root, m_shot_path_name, category_name))
        images_paths = sorted(glob.glob(category+'/*'))
        cnt=0
        times = -1
        img_batch = torch.zeros(batch_size, 3, 224,224).cuda()
        for img_path in images_paths:
            # img = cv2.imread(img_path)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # img = Image.fromarray(np.uint8(img))

            img = Image.open(img_path)
            img = TRAIN_TRANSFORMS(img)
            img = img.cuda()
            # img = torch.unsqueeze(img,dim=0)
            if img.shape[0] ==4:
                # import ipdb
                # ipdb.set_trace(context=20)
                img = img[:3]
            img_batch[cnt] = img
            cnt+=1

            del img
            torch.cuda.empty_cache()


            if cnt == batch_size:

                times+=1
                cnt=0
                _, feat = teacher_model(img_batch)

                for i in range(batch_size):
                    img_path_name = images_paths[times*batch_size+i]
                    entropy_dict[img_path_name] = feat[i].detach().cpu().numpy()

                del img_batch, feat
                torch.cuda.empty_cache()
                img_batch = torch.zeros(batch_size, 3, 224, 224).cuda()

        # if not divided by batch_size
        if cnt!=0:
            # import ipdb
            # ipdb.set_trace(context=20)
            img_batch = img_batch[:cnt]

            _, feat = teacher_model(img_batch)

            for i in range(cnt):
                img_path_name = images_paths[times * batch_size + i]
                entropy_dict[img_path_name] = feat[i].detach().cpu().numpy()

            del img_batch, feat
            torch.cuda.empty_cache()


        sorted_list = sorted(entropy_dict.items(), key=lambda x: x[0], reverse=True)

        # 1.get dis matrix
        num=len(sorted_list)
        dis_matrix = np.zeros((num,num))
        for i in range(num):
            for j in range(num):
                dis_matrix[i, j] = np.linalg.norm(sorted_list[i][1] - sorted_list[j][1])
                # print(dis_matrix[i, j])
            if i % 50 == 0:
                print('Counting {0}/{1}'.format(i + 1, num))

        # 2.greedy select and get index
        def getGreedyPerm(D, num):
            N = D.shape[0]
            # By default, takes the first point in the list to be the
            # first point in the permutation, but could be random
            perm = np.zeros(num, dtype=np.int64)
            ds = D[0, :]
            for i in range(num):
                idx = np.argmax(ds)
                perm[i] = idx
                ds = np.minimum(ds, D[idx, :])
                if i % 50 == 0:
                    print('getGreedyPerm {0}/{1}'.format(i + 1, num))
            return perm

        subset_index = getGreedyPerm(dis_matrix, m)

        # 3.index to names
        selected_imgs = [ sorted_list[i][0] for i in subset_index]

        # endtime = time.asctime(time.localtime(time.time()))
        # import ipdb
        # ipdb.set_trace(context=20)
        for selected in selected_imgs:
            selected_path = selected
            os.system('cp {} {}/../{}/{}/'.format(selected_path, data_root, m_shot_path_name, category_name))

        # endtime = time.asctime(time.localtime(time.time()))
        # print("start: ", starttime)
        # print("end: ", endtime)
        # import ipdb
        # ipdb.set_trace(context=20)

        # a=1


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

def get_resnet20_as_gift_and_load_model(args, ds, finetuned_model_path=None, fc_classes=None):
    '''Given arguments, dataset object and a finetuned model_path, returns a model
    with loaded weights and returns the checkpoint necessary for resuming training.
    '''
    def load_pretrained_model(model, finetuned_model_path):
        if args.abandom_fc:
            print('[Resuming finetuning from a resnet20 checkpoint abandoming original fc...]')
        else:
            print('[Resuming finetuning from a resnet20 checkpoint with original fc...]')
        print('loading from %s'%finetuned_model_path)
        resource = torch.load(finetuned_model_path)
        # import ipdb
        # ipdb.set_trace(context=20)
        if 'model' in resource.keys():
            pretrained_dict = resource['model']
        elif 'snet' in resource.keys():
            pretrained_dict = resource['snet']
        elif 'net' in resource.keys():
            pretrained_dict = resource['net']
        else:
            pretrained_dict = resource

        model_dict = model.state_dict()


        # 1. filter out unnecessary keys
        from collections import OrderedDict
        new_dict = OrderedDict()
        for k,v in pretrained_dict.items():
            if 'module.attacker.model.model' in k:
                new_dict[k.replace('module.attacker.model.model', '')] = v
            elif 'module.model.model' in k:
                new_dict[k.replace('module.model.model', '')] = v
            else:
                new_dict[k] = v
        import ipdb
        ipdb.set_trace(context=20)
        if args.abandom_fc or args.teacher_path: # if KD, also abandom student fc
            new_dict = {k: v for k, v in new_dict.items() if k in model_dict.keys() and 'fc' not in k}
        else:
            new_dict = {k: v for k, v in new_dict.items() if k in model_dict.keys()}
        # 2. overwrite entries in the existing state dict
        assert new_dict is not None, 'error in state dict name'
        model_dict.update(new_dict)
        # 3. load the new state dict
        model.load_state_dict(model_dict, strict=False)
    if args.dataset in list(transfer_datasets.DS_TO_FUNC.keys()) and not args.cifar10_cifar10:
        from resnet import resnet20_as_gift
        model = resnet20_as_gift(num_classes=ds.num_classes)
        # if finetuned_model_path:
        #     load_pretrained_model(model, finetuned_model_path)
        while hasattr(model, 'model'):
            model = model.model
        while hasattr(model, 'module'):
            model = model.module
        while hasattr(model, 'model'):
            model = model.model
        while hasattr(model, 'module'):
            model = model.module
        if args.abandom_fc:
            model = fine_tunify.ft(args.arch, model, ds.num_classes, args.additional_hidden)
        checkpoint = None
        # model, checkpoint = model_utils.make_and_restore_model(arch=model, dataset=ds, resume_path=finetuned_model_path,
        #                                                        add_custom_forward=args.additional_hidden > 0 or args.arch in pytorch_models.keys())
        if fc_classes is None:
            model, _ = model_utils.make_and_restore_model(arch=model, dataset=datasets.ImageNet(''), add_custom_forward=True, resume_path=finetuned_model_path)
        else:    # 5.1 UKD with 1000D fc
            model = fine_tunify.ft(
                args.arch, model, fc_classes, args.additional_hidden)
            model, _ = model_utils.make_and_restore_model(arch=model, dataset=datasets.ImageNet(''),
                                                          add_custom_forward=True)
    else:
        raise ValueError

    return model, checkpoint

def get_model(args, ds, fc_classes=None):
    '''Given arguments and a dataset object, returns an ImageNet model (with appropriate last layer changes to 
    fit the target dataset) and a checkpoint.The checkpoint is set to None if noe resuming training.
    '''
    finetuned_model_path = os.path.join(
        args.out_dir, args.exp_name, 'checkpoint.pt.latest')
    if args.arch == 'resnet20_as_gift': # TODO: only support ckpt trained from the same dataset (fc)
        model, checkpoint = get_resnet20_as_gift_and_load_model(
            args, ds, args.weight_path, fc_classes)
    elif args.resume and os.path.isfile(finetuned_model_path):
        model, checkpoint = resume_finetuning_from_checkpoint(
            args, ds, finetuned_model_path)
    elif args.weight and os.path.isfile(args.weight_path):
        model, checkpoint = resume_finetuning_from_checkpoint(
            args, ds, args.weight_path)
        checkpoint = None
    else:
        if args.dataset in list(transfer_datasets.DS_TO_FUNC.keys()) and not args.cifar10_cifar10 and not args.resume_from_fc_of_target_data:
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

            if not freeze and f'layer{freeze_level}' not in name:
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
        ['imagenet', 'breeds_living_9', 'stylized_imagenet']
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
        'mobilenet': models.mobilenet_v2,
        'resnext50_32x4d': models.resnext50_32x4d,
        'mnasnet': models.mnasnet1_0,
        'resnet34': models.resnet34,
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
