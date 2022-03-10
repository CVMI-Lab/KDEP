import torch as ch
import numpy as np
import torch.nn as nn
from torch.optim import SGD, lr_scheduler
from torchvision.utils import make_grid
from cox.utils import Parameters

from robustness.tools import helpers
from robustness.tools.helpers import AverageMeter, ckpt_at_epoch, has_attr
from robustness.tools import constants as consts
import dill
import os
import time
import warnings
from KD import DistillKL, AngleLoss, DistillKL_mask
import torch.nn.functional as F

if int(os.environ.get("NOTEBOOK_MODE", 0)) == 1:
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm as tqdm

try:
    from apex import amp
except Exception as e:
    warnings.warn('Could not import amp.')


def L2_loss_FKD(x, y, norm=False, exp_mode='exp', T=16, align=False):
    '''
    Compute L2 between two tensors
    '''
    # x: N x D
    # y: N x D

    if norm:
        x_n, y_n = x.norm(p=2, dim=1, keepdim=True), y.norm(p=2, dim=1, keepdim=True)
        x = x / (x_n.expand_as(x))
        y = y / (y_n.expand_as(y))

    if align:
        # import ipdb
        # ipdb.set_trace(context=20)
        cos_sim = F.cosine_similarity(x,y) # B
        x=x.permute(1,0)
        x = ch.sign(cos_sim) * x
        x = x.permute(1, 0)

    if exp_mode == 'exp':
        x = ch.exp(x / T)
        y = ch.exp(y / T)
    elif exp_mode == 'softmax':
        x = ch.nn.functional.softmax(x / T, dim=1)
        y = ch.nn.functional.softmax(y / T, dim=1)
        return ch.nn.functional.mse_loss(x, y, reduction='sum')

    return ch.nn.functional.mse_loss(x, y, reduction='mean')

def check_required_args(args, eval_only=False):
    """
    Check that the required training arguments are present.

    Args:
        args (argparse object): the arguments to check
        eval_only (bool) : whether to check only the arguments for evaluation
    """
    required_args_eval = ["adv_eval"]
    required_args_train = ["epochs", "out_dir", "adv_train",
                           "log_iters", "lr", "momentum", "weight_decay"]
    adv_required_args = ["attack_steps", "eps", "constraint",
                         "use_best", "attack_lr", "random_restarts"]

    # Generic function for checking all arguments in a list
    def check_args(args_list):
        for arg in args_list:
            assert has_attr(args, arg), f"Missing argument {arg}"

    # Different required args based on training or eval:
    if not eval_only:
        check_args(required_args_train)
    else:
        check_args(required_args_eval)
    # More required args if we are robustly training or evaling
    is_adv = bool(args.adv_train) or bool(args.adv_eval)
    if is_adv:
        check_args(adv_required_args)
    # More required args if the user provides a custom training loss
    has_custom_train = has_attr(args, 'custom_train_loss')
    has_custom_adv = has_attr(args, 'custom_adv_loss')
    if has_custom_train and is_adv and not has_custom_adv:
        raise ValueError("Cannot use custom train loss \
            without a custom adversarial loss (see docs)")


def make_optimizer_and_schedule(args, model, checkpoint, params):
    """
    *Internal Function* (called directly from train_model)

    Creates an optimizer and a schedule for a given model, restoring from a
    checkpoint if it is non-null.

    Args:
        args (object) : an arguments object, see
            :meth:`~robustness.train.train_model` for details
        model (AttackerModel) : the model to create the optimizer for
        checkpoint (dict) : a loaded checkpoint saved by this library and loaded
            with `ch.load`
        params (list|None) : a list of parameters that should be updatable, all
            other params will not update. If ``None``, update all params

    Returns:
        An optimizer (ch.nn.optim.Optimizer) and a scheduler
            (ch.nn.optim.lr_schedulers module).
    """
    # Make optimizer
    param_list = model.parameters() if params is None else params
    if args.optimizer_custom == 'sgd':
        optimizer = SGD(param_list, args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer_custom == 'adabelief':
        # pip install adabelief-pytorch==0.2.0
        from adabelief_pytorch import AdaBelief
        optimizer = AdaBelief(param_list, lr=args.lr, eps=1e-8, betas=(0.9, 0.999), weight_decouple=True, weight_decay=args.weight_decay,rectify=False)
    else:
        optimizer = None

    if args.mixed_precision:
        model.to('cuda')
        model, optimizer = amp.initialize(model, optimizer, 'O1')

    # Make schedule
    schedule = None
    if args.custom_lr_schedule == 'cyclic':
        eps = args.epochs
        lr_func = lambda t: np.interp([t], [0, eps * 4 // 15, eps], [0, 1, 0])[0]
        schedule = lr_scheduler.LambdaLR(optimizer, lr_func)
    elif args.custom_lr_schedule == '1cycle':
        eps = args.epochs
        max_lr = args.max_lr
        # lr_func = lambda t: np.interp([t], [0, eps // 2, eps], [1, max_lr // args.lr, 1])[0]
        lr_func = lambda t: np.interp([t], [0, eps // 5 * 2, eps // 5 * 4, eps], [1, max_lr / args.lr, 1, 5e-5/args.lr])[0]
        schedule = lr_scheduler.LambdaLR(optimizer, lr_func)
    elif args.custom_lr_schedule == '3cycle':
        eps = args.epochs
        max_lr = args.max_lr
        # lr_func = lambda t: np.interp([t], [0, eps // 2, eps], [1, max_lr // args.lr, 1])[0]  # args.lr=0.03 or 0.05
        lr_func = lambda t: np.interp([t], [0, eps // 6 * 1, eps // 6 * 2, eps // 6 * 3, eps // 6 * 4, eps // 6 * 5, eps],
                                      [1, max_lr / args.lr, 1/10, 10, 1/100, 1, 5e-5/args.lr])[0]
        schedule = lr_scheduler.LambdaLR(optimizer, lr_func)
    elif args.custom_lr_schedule == 'multisteplr':
        args.milestones = [int(i) for i in args.milestones]
        schedule = lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.1)
    elif args.custom_lr_schedule:
        cs = args.custom_lr_schedule
        periods = eval(cs) if type(cs) is str else cs
        if args.lr_interpolation == 'linear':
            lr_func = lambda t: np.interp([t], *zip(*periods))[0]
        else:
            def lr_func(ep):
                for (milestone, lr) in reversed(periods):
                    if ep >= milestone: return lr
                return 1.0
        schedule = lr_scheduler.LambdaLR(optimizer, lr_func)
    elif args.step_lr:
        schedule = lr_scheduler.StepLR(optimizer, step_size=args.step_lr, gamma=args.step_lr_gamma)

    # Fast-forward the optimizer and the scheduler if resuming
    if checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        try:
            schedule.load_state_dict(checkpoint['schedule'])
        except:
            steps_to_take = checkpoint['epoch']
            print('Could not load schedule (was probably LambdaLR).'
                  f' Stepping {steps_to_take} times instead...')
            for i in range(steps_to_take):
                schedule.step()

        if 'amp' in checkpoint and checkpoint['amp'] not in [None, 'N/A']:
            amp.load_state_dict(checkpoint['amp'])

        # TODO: see if there's a smarter way to do this
        # TODO: see what's up with loading fp32 weights and then MP training
        if args.mixed_precision:
            model.load_state_dict(checkpoint['model'])

    return optimizer, schedule


def eval_model(args, model, loader, store, warp_robust_code=True, all_feat=False):
    """
    Evaluate a model for standard (and optionally adversarial) accuracy.

    Args:
        args (object) : A list of arguments---should be a python object
            implementing ``getattr()`` and ``setattr()``.
        model (AttackerModel) : model to evaluate
        loader (iterable) : a dataloader serving `(input, label)` batches from
            the validation set
        store (cox.Store) : store for saving results in (via tensorboardX)
    """
    loop_choice = _model_loop if warp_robust_code else _model_loop_clean
    check_required_args(args, eval_only=True)
    start_time = time.time()

    if store is not None:
        store.add_table(consts.LOGS_TABLE, consts.LOGS_SCHEMA)
    writer = store.tensorboard if store else None

    assert not hasattr(model, "module"), "model is already in DataParallel."
    model = ch.nn.DataParallel(model).cuda()

    # prec1, nat_loss = loop_choice(args, 'val', loader, model, None, 0, False, writer, all_feat=all_feat)
    prec1, nat_loss = loop_choice(args, 'val', loader, model, None, 0, False, writer)

    adv_prec1, adv_loss = float('nan'), float('nan')
    if args.adv_eval:
        args.eps = eval(str(args.eps)) if has_attr(args, 'eps') else None
        args.attack_lr = eval(str(args.attack_lr)) if has_attr(args, 'attack_lr') else None
        adv_prec1, adv_loss = loop_choice(args, 'val', loader,
                                          model, None, 0, True, writer)
    log_info = {
        'epoch': 0,
        'nat_prec1': prec1,
        'adv_prec1': adv_prec1,
        'nat_loss': nat_loss,
        'adv_loss': adv_loss,
        'train_prec1': float('nan'),
        'train_loss': float('nan'),
        'time': time.time() - start_time
    }

    # Log info into the logs table
    if store: store[consts.LOGS_TABLE].append_row(log_info)
    return log_info


def train_model(args, model, loaders, *, checkpoint=None, dp_device_ids=None,
                store=None, update_params=None, disable_no_grad=False, teacher_model=None, warp_robust_code=True, all_feat=False):
    """
    Main function for training a model.

    Args:
        args (object) : A python object for arguments, implementing
            ``getattr()`` and ``setattr()`` and having the following
            attributes. See :attr:`robustness.defaults.TRAINING_ARGS` for a
            list of arguments, and you can use
            :meth:`robustness.defaults.check_and_fill_args` to make sure that
            all required arguments are filled and to fill missing args with
            reasonable defaults:

            adv_train (int or bool, *required*)
                if 1/True, adversarially train, otherwise if 0/False do
                standard training
            epochs (int, *required*)
                number of epochs to train for
            lr (float, *required*)
                learning rate for SGD optimizer
            weight_decay (float, *required*)
                weight decay for SGD optimizer
            momentum (float, *required*)
                momentum parameter for SGD optimizer
            step_lr (int)
                if given, drop learning rate by 10x every `step_lr` steps
            custom_lr_multplier (str)
                If given, use a custom LR schedule, formed by multiplying the
                    original ``lr`` (format: [(epoch, LR_MULTIPLIER),...])
            lr_interpolation (str)
                How to drop the learning rate, either ``step`` or ``linear``,
                    ignored unless ``custom_lr_schedule`` is provided.
            adv_eval (int or bool)
                If True/1, then also do adversarial evaluation, otherwise skip
                (ignored if adv_train is True)
            log_iters (int, *required*)
                How frequently (in epochs) to save training logs
            save_ckpt_iters (int, *required*)
                How frequently (in epochs) to save checkpoints (if -1, then only
                save latest and best ckpts)
            attack_lr (float or str, *required if adv_train or adv_eval*)
                float (or float-parseable string) for the adv attack step size
            constraint (str, *required if adv_train or adv_eval*)
                the type of adversary constraint
                (:attr:`robustness.attacker.STEPS`)
            eps (float or str, *required if adv_train or adv_eval*)
                float (or float-parseable string) for the adv attack budget
            attack_steps (int, *required if adv_train or adv_eval*)
                number of steps to take in adv attack
            custom_eps_multiplier (str, *required if adv_train or adv_eval*)
                If given, then set epsilon according to a schedule by
                multiplying the given eps value by a factor at each epoch. Given
                in the same format as ``custom_lr_schedule``, ``[(epoch,
                MULTIPLIER)..]``
            use_best (int or bool, *required if adv_train or adv_eval*) :
                If True/1, use the best (in terms of loss) PGD step as the
                attack, if False/0 use the last step
            random_restarts (int, *required if adv_train or adv_eval*)
                Number of random restarts to use for adversarial evaluation
            custom_train_loss (function, optional)
                If given, a custom loss instead of the default CrossEntropyLoss.
                Takes in `(logits, targets)` and returns a scalar.
            custom_adv_loss (function, *required if custom_train_loss*)
                If given, a custom loss function for the adversary. The custom
                loss function takes in `model, input, target` and should return
                a vector representing the loss for each element of the batch, as
                well as the classifier output.
            custom_accuracy (function)
                If given, should be a function that takes in model outputs
                and model targets and outputs a top1 and top5 accuracy, will
                displayed instead of conventional accuracies
            regularizer (function, optional)
                If given, this function of `model, input, target` returns a
                (scalar) that is added on to the training loss without being
                subject to adversarial attack
            iteration_hook (function, optional)
                If given, this function is called every training iteration by
                the training loop (useful for custom logging). The function is
                given arguments `model, iteration #, loop_type [train/eval],
                current_batch_ims, current_batch_labels`.
            epoch hook (function, optional)
                Similar to iteration_hook but called every epoch instead, and
                given arguments `model, log_info` where `log_info` is a
                dictionary with keys `epoch, nat_prec1, adv_prec1, nat_loss,
                adv_loss, train_prec1, train_loss`.

        model (AttackerModel) : the model to train.
        loaders (tuple[iterable]) : `tuple` of data loaders of the form
            `(train_loader, val_loader)`
        checkpoint (dict) : a loaded checkpoint previously saved by this library
            (if resuming from checkpoint)
        dp_device_ids (list|None) : if not ``None``, a list of device ids to
            use for DataParallel.
        store (cox.Store) : a cox store for logging training progress
        update_params (list) : list of parameters to use for training, if None
            then all parameters in the model are used (useful for transfer
            learning)
        disable_no_grad (bool) : if True, then even model evaluation will be
            run with autograd enabled (otherwise it will be wrapped in a ch.no_grad())
    """
    # Logging setup
    loop_choice = _model_loop if warp_robust_code else _model_loop_clean
    writer = store.tensorboard if store else None
    prec1_key = f"{'adv' if args.adv_train else 'nat'}_prec1"
    if store is not None:
        store.add_table(consts.LOGS_TABLE, consts.LOGS_SCHEMA)

    # Reformat and read arguments
    check_required_args(args)  # Argument sanity check
    for p in ['eps', 'attack_lr', 'custom_eps_multiplier']:
        setattr(args, p, eval(str(getattr(args, p))) if has_attr(args, p) else None)
    if args.custom_eps_multiplier is not None:
        eps_periods = args.custom_eps_multiplier
        args.custom_eps_multiplier = lambda t: np.interp([t], *zip(*eps_periods))[0]

    # Initial setup
    train_loader, val_loader = loaders
    opt, schedule = make_optimizer_and_schedule(args, model, checkpoint, update_params)

    # Put the model into parallel mode
    assert not hasattr(model, "module"), "model is already in DataParallel."
    model = ch.nn.DataParallel(model, device_ids=dp_device_ids).cuda() # todo: teacher, not need, already done when evaling

    if teacher_model:
        assert not hasattr(teacher_model, "module"), "model is already in DataParallel."
        teacher_model = ch.nn.DataParallel(teacher_model, device_ids=dp_device_ids).cuda()

    best_prec1, start_epoch = (0, 0)
    best_feat_relation_dis = 1000000
    best_logit_relation_dis = 1000000
    if checkpoint:
        start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint[prec1_key] if prec1_key in checkpoint \
            else loop_choice(args, 'val', val_loader, model, None, start_epoch - 1, args.adv_train, writer=None, all_feat=all_feat)[0]

    # Timestamp for training start time
    start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        # if args.KD_r>0 and epoch == 30:
        #     args.KD_r = 100
        # train for one epoch
        train_prec1, train_loss = loop_choice(args, 'train', train_loader,
                                              model, opt, epoch, args.adv_train, writer, teacher_model=teacher_model, all_feat=all_feat)
        last_epoch = (epoch == (args.epochs - 1))

        # evaluate on validation set
        sd_info = {
            'model': model.state_dict(),
            'optimizer': opt.state_dict(),
            'schedule': (schedule and schedule.state_dict()),
            'epoch': epoch + 1,
            'amp': amp.state_dict() if args.mixed_precision else None,
        }

        def save_checkpoint(filename):
            ckpt_save_path = os.path.join(args.out_dir if not store else \
                                              store.path, filename)
            ch.save(sd_info, ckpt_save_path, pickle_module=dill)

        save_its = args.save_ckpt_iters
        should_save_ckpt = (epoch % save_its == 0) and (save_its > 0)
        should_log = (epoch % args.log_iters == 0)

        if should_log or last_epoch or should_save_ckpt:
            # log + get best
            ctx = ch.enable_grad() if disable_no_grad else ch.no_grad()
            with ctx:
                # prec1, nat_loss = loop_choice(args, 'val', val_loader, model,
                #                               None, epoch, False, writer, all_feat=all_feat)

                if args.epochs <200:
                    prec1, nat_loss = loop_choice(args, 'val', val_loader, model,
                                                                 None, epoch, False, writer,
                                                                 teacher_model=teacher_model, all_feat=all_feat)
                else:
                    prec1, nat_loss = 0, 0

            # loader, model, epoch, input_adv_exs
            should_adv_eval = args.adv_eval or args.adv_train
            adv_val = should_adv_eval and loop_choice(args, 'val', val_loader,
                                                      model, None, epoch, True, writer, all_feat=all_feat)
            adv_prec1, adv_loss = adv_val or (-1.0, -1.0)

            # remember best prec@1 and save checkpoint
            our_prec1 = adv_prec1 if args.adv_train else prec1
            is_best = our_prec1 > best_prec1
            best_prec1 = max(our_prec1, best_prec1)
            sd_info[prec1_key] = our_prec1

            # log every checkpoint
            log_info = {
                'epoch': epoch + 1,
                'nat_prec1': prec1,
                'adv_prec1': adv_prec1,
                'nat_loss': nat_loss,
                'adv_loss': adv_loss,
                'train_prec1': train_prec1,
                'train_loss': train_loss,
                'time': time.time() - start_time
            }

            # Log info into the logs table
            if store: store[consts.LOGS_TABLE].append_row(log_info)
            # If we are at a saving epoch (or the last epoch), save a checkpoint
            if should_save_ckpt or last_epoch: save_checkpoint(ckpt_at_epoch(epoch))

            # Update the latest and best checkpoints (overrides old one)
            save_checkpoint(consts.CKPT_NAME_LATEST)
            if is_best: save_checkpoint(consts.CKPT_NAME_BEST)

        # import ipdb
        # ipdb.set_trace(context=20)
        if args.save_epoch_list and str(epoch+1) in args.save_epoch_list: save_checkpoint(str(epoch+1)+'.ckpt')

        if schedule: schedule.step()
        if has_attr(args, 'epoch_hook'): args.epoch_hook(model, log_info)

    return model


def _model_loop(args, loop_type, loader, model, opt, epoch, adv, writer, teacher_model=None, all_feat=False):
    """
    *Internal function* (refer to the train_model and eval_model functions for
    how to train and evaluate models).

    Runs a single epoch of either training or evaluating.

    Args:
        args (object) : an arguments object (see
            :meth:`~robustness.train.train_model` for list of arguments
        loop_type ('train' or 'val') : whether we are training or evaluating
        loader (iterable) : an iterable loader of the form
            `(image_batch, label_batch)`
        model (AttackerModel) : model to train/evaluate
        opt (ch.optim.Optimizer) : optimizer to use (ignored for evaluation)
        epoch (int) : which epoch we are currently on
        adv (bool) : whether to evaluate adversarially (otherwise standard)
        writer : tensorboardX writer (optional)
        teacher_model: knowledge distillation teacher

    Returns:
        The average top1 accuracy and the average loss across the epoch.
    """
    if not loop_type in ['train', 'val']:
        err_msg = "loop_type ({0}) must be 'train' or 'val'".format(loop_type)
        raise ValueError(err_msg)
    is_train = (loop_type == 'train')

    losses = AverageMeter()
    loss_KD_feat_meter = AverageMeter()
    loss_KD_feat_mid_meter = AverageMeter()
    loss_KD_feat_rela_meter = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    grad_norm_meter = AverageMeter()


    prec = 'NatPrec' if not adv else 'AdvPrec'
    loop_msg = 'Train' if loop_type == 'train' else 'Val'

    # switch to train/eval mode depending
    model = model.train() if is_train else model.eval()


    # If adv training (or evaling), set eps and random_restarts appropriately
    if adv:
        eps = args.custom_eps_multiplier(epoch) * args.eps \
            if (is_train and args.custom_eps_multiplier) else args.eps
        random_restarts = 0 if is_train else args.random_restarts

    # Custom training criterion
    has_custom_train_loss = has_attr(args, 'custom_train_loss')
    train_criterion = args.custom_train_loss if has_custom_train_loss \
        else ch.nn.CrossEntropyLoss()

    if teacher_model:
        teacher_model.eval()
        KD_criterion = DistillKL(args.KD_T)
        KD_criterion_mid = DistillKL(args.KD_T_mid)
        # KD_criterion = DistillKL_mask(args.KD_T, args.threshold, args.fc_classes)

    angle_loss_criterion = AngleLoss()

    has_custom_adv_loss = has_attr(args, 'custom_adv_loss')
    adv_criterion = args.custom_adv_loss if has_custom_adv_loss else None

    attack_kwargs = {}
    if adv:
        attack_kwargs = {
            'constraint': args.constraint,
            'eps': eps,
            'step_size': args.attack_lr,
            'iterations': args.attack_steps,
            'random_start': args.random_start,
            'custom_loss': adv_criterion,
            'random_restarts': random_restarts,
            'use_best': bool(args.use_best)
        }

    iterator = tqdm(enumerate(loader), total=len(loader))
    current_lr = opt.state_dict()['param_groups'][0]['lr'] if opt is not None else 0

    for i, (inp, target) in iterator:
        # measure data loading time
        # import ipdb
        # ipdb.set_trace(context=20)
        target = target.cuda(non_blocking=True)
        # inp = inp.cuda(non_blocking=True)
        output, final_inp = model(inp, target=target, make_adv=adv,
                                  **attack_kwargs)

        loss_KD_feat_mid = 0 if args.multi else ch.zeros(1).cuda()

        if teacher_model:
            # teacher_model = teacher_model.cuda(non_blocking=True)
            output_teacher, _ = teacher_model(inp, target=target, make_adv=adv,
                                      **attack_kwargs)
            student_logits = output[0] if (type(output) is tuple) else output
            teacher_logits = output_teacher[0] if (type(output_teacher) is tuple) else output_teacher
            if args.KD_a >0:
                # import ipdb
                # ipdb.set_trace(context=20)
                loss_KD_logits = KD_criterion(student_logits, teacher_logits)
            else:
                loss_KD_logits = ch.zeros(1).cuda()
            # loss = (1-args.KD_a) * loss_cls + args.KD_a * loss_KD

            if args.loss_w >0:
                feat_s = output[1]
                feat_t = output_teacher[1]
                loss_KD_feat = L2_loss_FKD(feat_s, feat_t, norm=False, exp_mode='none', T=args.KD_T) # L2 loss for feat KD

            else:
                loss_KD_feat = ch.zeros(1).cuda()
            loss_KD_feat_meter.update(args.loss_w * loss_KD_feat.item())
            loss_KD_feat_mid_meter.update(args.KD_multi * args.loss_w * loss_KD_feat_mid.item())
            if args.KD_c >0:
                loss_cls = train_criterion(student_logits, target)
            else:
                loss_cls = ch.zeros(1).cuda()
            loss = args.KD_a * loss_KD_logits + args.loss_w * loss_KD_feat + args.KD_c * loss_cls


        else:
            output = output[0] if (type(output) is tuple) else output
            loss = train_criterion(output, target)

        if len(loss.shape) > 0: loss = loss.mean()

        model_logits = output[0] if (type(output) is tuple) else output

        # measure accuracy and record loss
        top1_acc = float('nan')
        top5_acc = float('nan')
        try:
            maxk = min(5, model_logits.shape[-1])
            if has_attr(args, "custom_accuracy"):
                prec1, prec5 = args.custom_accuracy(model_logits, target)
            else:
                prec1, prec5 = helpers.accuracy(model_logits, target, topk=(1, maxk))
                # import ipdb
                # ipdb.set_trace(context=20)
                prec1, prec5 = prec1[0], prec5[0]

            losses.update(loss.item(), inp.size(0))
            top1.update(prec1, inp.size(0))
            top5.update(prec5, inp.size(0))

            top1_acc = top1.avg
            top5_acc = top5.avg
        except:
            warnings.warn('Failed to calculate the accuracy.')

        reg_term = 0.0
        if has_attr(args, "regularizer"):
            reg_term = args.regularizer(model, inp, target)
        loss = loss + reg_term

        # compute gradient and do SGD step
        if is_train:
            opt.zero_grad()
            if args.mixed_precision:
                with amp.scale_loss(loss, opt) as sl:
                    sl.backward()
            else:
                loss.backward()

            opt.step()
        elif adv and i == 0 and writer:
            # add some examples to the tensorboard
            nat_grid = make_grid(inp[:15, ...])
            adv_grid = make_grid(final_inp[:15, ...])
            writer.add_image('Nat input', nat_grid, epoch)
            writer.add_image('Adv input', adv_grid, epoch)

        # ITERATOR

        # if has_attr(model.module.model.model, "feat_scale"):
        #     feat_scale = model.module.model.model.feat_scale
        # else:
        feat_scale = 1

        desc = ('{2} Epoch:{0} | Loss {loss.avg:.4f} | '
                '{1}1 {top1_acc:.3f} | {1}5 {top5_acc:.3f} | '
                'Loss_feat: {feat:.3f} |'
                'Loss_feat_mid: {feat_mid:.3f} |'
                'Loss_feat_rela: {loss_feat_rela:.3f} |'
                # 'grad norm avg: {grad_norm:.3f}|'
                # 'grad norm val: {grad_norm_val:.3f}|'
                # 'KD_r: {KD_r}'
                'feat_scale: {fs:.3f}'
                'lr: {lr}||'.format(epoch, prec, loop_msg,
                                            loss=losses, top1_acc=top1_acc, top5_acc=top5_acc, feat=loss_KD_feat_meter.val, feat_mid=loss_KD_feat_mid_meter.val, loss_feat_rela=loss_KD_feat_rela_meter.val,
                                    grad_norm=grad_norm_meter.avg,grad_norm_val=grad_norm_meter.val,KD_r=args.KD_r, fs=feat_scale, lr=current_lr))

        if i == len(loader)-1:
            print('{2} Epoch:{0} | Loss {loss.avg:.4f} | '
                '{1}1 {top1_acc:.3f} | {1}5 {top5_acc:.3f} | '
                'Loss_feat: {feat:.3f} |'
                'Loss_feat_mid: {feat_mid:.3f} |'
                'Loss_feat_rela: {loss_feat_rela:.3f} |'
                # 'grad norm avg: {grad_norm:.3f}|'
                # 'grad norm val: {grad_norm_val:.3f}|'
                # 'KD_r: {KD_r}'
                'feat_scale: {fs:.3f}'
                'lr: {lr}||'.format(epoch, prec, loop_msg,
                                            loss=losses, top1_acc=top1_acc, top5_acc=top5_acc, feat=loss_KD_feat_meter.val, feat_mid=loss_KD_feat_mid_meter.val, loss_feat_rela=loss_KD_feat_rela_meter.val,
                                    grad_norm=grad_norm_meter.avg,grad_norm_val=grad_norm_meter.val,KD_r=args.KD_r, fs=feat_scale, lr=current_lr))


        # USER-DEFINED HOOK
        if has_attr(args, 'iteration_hook'):
            args.iteration_hook(model, i, loop_type, inp, target)

        iterator.set_description(desc)
        iterator.refresh()

    if writer is not None:
        prec_type = 'adv' if adv else 'nat'
        descs = ['loss', 'top1', 'top5']
        vals = [losses, top1, top5]
        for d, v in zip(descs, vals):
            writer.add_scalar('_'.join([prec_type, loop_type, d]), v.avg,
                              epoch)

    # return top1.avg, losses.avg, feat_relation_dis_meter.avg, logit_relation_dis_meter.avg
    return top1.avg, losses.avg


def _model_loop_clean(args, loop_type, loader, model, opt, epoch, adv, writer, teacher_model=None, all_feat=False):
    """
    *Internal function* (refer to the train_model and eval_model functions for
    how to train and evaluate models).

    Runs a single epoch of either training or evaluating.

    Args:
        args (object) : an arguments object (see
            :meth:`~robustness.train.train_model` for list of arguments
        loop_type ('train' or 'val') : whether we are training or evaluating
        loader (iterable) : an iterable loader of the form
            `(image_batch, label_batch)`
        model (AttackerModel) : model to train/evaluate
        opt (ch.optim.Optimizer) : optimizer to use (ignored for evaluation)
        epoch (int) : which epoch we are currently on
        adv (bool) : whether to evaluate adversarially (otherwise standard)
        writer : tensorboardX writer (optional)
        teacher_model: knowledge distillation teacher

    Returns:
        The average top1 accuracy and the average loss across the epoch.
    """
    if not loop_type in ['train', 'val']:
        err_msg = "loop_type ({0}) must be 'train' or 'val'".format(loop_type)
        raise ValueError(err_msg)
    is_train = (loop_type == 'train')

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    prec = 'NatPrec' if not adv else 'AdvPrec'
    loop_msg = 'Train' if loop_type == 'train' else 'Val'

    # switch to train/eval mode depending
    model = model.train() if is_train else model.eval()


    # If adv training (or evaling), set eps and random_restarts appropriately
    if adv:
        eps = args.custom_eps_multiplier(epoch) * args.eps \
            if (is_train and args.custom_eps_multiplier) else args.eps
        random_restarts = 0 if is_train else args.random_restarts

    # Custom training criterion
    has_custom_train_loss = has_attr(args, 'custom_train_loss')
    train_criterion = args.custom_train_loss if has_custom_train_loss \
        else ch.nn.CrossEntropyLoss()

    if teacher_model:
        teacher_model.eval()
        KD_criterion = DistillKL(args.KD_T)

    has_custom_adv_loss = has_attr(args, 'custom_adv_loss')
    adv_criterion = args.custom_adv_loss if has_custom_adv_loss else None

    # FKD_criterion = ch.nn.functional.mse_loss(reduction='mean')
    from robustness.datasets import DataSet, CIFAR, ImageNet
    # from .utils import constants as cs
    from robustness import datasets
    # dataset = ImageNet(cs.IMGNET_PATH)
    dataset = datasets.ImageNet('')
    normalizer = helpers.InputNormalize(dataset.mean, dataset.std)
    # normalizer = helpers.InputNormalize(ch.tensor([0.5, 0.5, 0.5]), ch.tensor([0.5, 0.5, 0.5]))

    if all_feat:
        # FSP
        from KD import FSP
        # model.eval()
        # data = ch.randn(2, 3, 224, 224)
        # _, feat_s = model(data, all_feat=all_feat)
        # _, feat_t = teacher_model(data, all_feat=all_feat)
        # model = model.train() if is_train else model.eval()

        # s_shapes = [s.shape for s in feat_s]
        # t_shapes = [t.shape for t in feat_t]
        # criterion_FSP = FSP(s_shapes, t_shapes)

        criterion_FSP = FSP()

    iterator = tqdm(enumerate(loader), total=len(loader))
    for i, (inp, target) in iterator:
        # measure data loading time

        inp = normalizer(inp)
        target = target.cuda(non_blocking=True)
        output, feat_s = model(inp)
        # import ipdb
        # ipdb.set_trace(context=20)
        if teacher_model:
            # teacher_model = teacher_model.cuda(non_blocking=True)
            output_teacher, feat_t = teacher_model(inp, all_feat=all_feat)
            student_logits = output[0] if (type(output) is tuple) else output
            teacher_logits = output_teacher[0] if (type(output_teacher) is tuple) else output_teacher
            loss_KD_logits = KD_criterion(student_logits, teacher_logits)
            if args.loss_w >0:
                if all_feat is False:
                    loss_KD_feat = ch.nn.functional.mse_loss(feat_s, feat_t, reduction='mean')
                else:
                    # FSP
                    loss_group = criterion_FSP(feat_s, feat_t)
                    loss_KD_feat = sum(loss_group)
            else:
                loss_KD_feat = 0
            loss_cls = train_criterion(output, target)
            loss = args.KD_a * loss_KD_logits + args.loss_w * loss_KD_feat + args.KD_c * loss_cls



        else:
            loss = train_criterion(output, target)

        if len(loss.shape) > 0: loss = loss.mean()

        model_logits = output[0] if (type(output) is tuple) else output

        # measure accuracy and record loss
        top1_acc = float('nan')
        top5_acc = float('nan')
        try:
            maxk = min(5, model_logits.shape[-1])
            if has_attr(args, "custom_accuracy"):
                prec1, prec5 = args.custom_accuracy(model_logits, target)
            else:
                prec1, prec5 = helpers.accuracy(model_logits, target, topk=(1, maxk))

                prec1, prec5 = prec1[0], prec5[0]

            losses.update(loss.item(), inp.size(0))
            top1.update(prec1, inp.size(0))
            top5.update(prec5, inp.size(0))

            top1_acc = top1.avg
            top5_acc = top5.avg
        except:
            warnings.warn('Failed to calculate the accuracy.')

        reg_term = 0.0
        if has_attr(args, "regularizer"):
            reg_term = args.regularizer(model, inp, target)
        loss = loss + reg_term

        # compute gradient and do SGD step
        if is_train:
            opt.zero_grad()
            if args.mixed_precision:
                with amp.scale_loss(loss, opt) as sl:
                    sl.backward()
            else:
                loss.backward()
            opt.step()

        # ITERATOR
        desc = ('{2} Epoch:{0} | Loss {loss.avg:.4f} | '
                '{1}1 {top1_acc:.3f} | {1}5 {top5_acc:.3f} | '
                'Reg term: {reg} ||'.format(epoch, prec, loop_msg,
                                            loss=losses, top1_acc=top1_acc, top5_acc=top5_acc, reg=reg_term))

        if i == len(loader)-1:
            print('{2} Epoch:{0} | Loss {loss.avg:.4f} | '
              '{1}1 {top1_acc:.3f} | {1}5 {top5_acc:.3f} | '
              'Reg term: {reg} ||'.format(epoch, prec, loop_msg,
                                          loss=losses, top1_acc=top1_acc, top5_acc=top5_acc, reg=reg_term))

        # USER-DEFINED HOOK
        if has_attr(args, 'iteration_hook'):
            args.iteration_hook(model, i, loop_type, inp, target)

        iterator.set_description(desc)
        iterator.refresh()

    if writer is not None:
        prec_type = 'adv' if adv else 'nat'
        descs = ['loss', 'top1', 'top5']
        vals = [losses, top1, top5]
        for d, v in zip(descs, vals):
            writer.add_scalar('_'.join([prec_type, loop_type, d]), v.avg,
                              epoch)

    return top1.avg, losses.avg


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
        # self.sum += val
        self.count += n
        self.avg = self.sum / self.count