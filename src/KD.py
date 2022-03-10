# ---- adpot from https://github.com/HobbitLong/RepDistiller/blob/master/distiller_zoo/KD.py

from __future__ import print_function

import torch
import torch as ch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]
        return loss

class DistillKL_mask(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T, thres, num_class):
        super(DistillKL_mask, self).__init__()
        self.thres = thres
        self.T = T
        self.num_class = num_class

    def forward(self, y_s, y_t, logit_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        # import ipdb
        # ipdb.set_trace(context=20)
        loss = F.kl_div(p_s, p_t, reduction='none') * (self.T**2) / y_s.shape[0] # B 512
        loss = loss.sum(1) # B

        ent = -(logit_t.softmax(1) * logit_t.log_softmax(1)).sum(1) # B
        threshold = self.thres * math.log(self.num_class)
        mask = ent.le(threshold).float()
        loss = (loss * mask).sum()

        return loss

class AngleLoss(nn.Module):
    """arccos(f_s * f_t / ||f_s|| * ||f_t||)"""
    def __init__(self):
        super(AngleLoss, self).__init__()

    def forward(self, a, b):
        eps = 1e-8
        a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]

        a_b = ch.einsum('bs,bs->b', a, b)
        a_b_norm = ch.einsum('b,b->b', a_n.squeeze(), b_n.squeeze())

        target = a_b / (a_b_norm+eps)
        loss_each = ch.acos(target)
        #loss_each = loss_each ** 2
        loss = loss_each.mean()
        return loss

class FSP(nn.Module):
    """A Gift from Knowledge Distillation:
    Fast Optimization, Network Minimization and Transfer Learning"""
    # def __init__(self, s_shapes, t_shapes):
    def __init__(self):
        super(FSP, self).__init__()
        # assert len(s_shapes) == len(t_shapes), 'unequal length of feat list'
        # s_c = [s[1] for s in s_shapes]
        # t_c = [t[1] for t in t_shapes]
        # if np.any(np.asarray(s_c) != np.asarray(t_c)):
        #     raise ValueError('num of channels not equal (error in FSP)')

    def forward(self, g_s, g_t):
        s_fsp = self.compute_fsp(g_s)
        t_fsp = self.compute_fsp(g_t)
        # import ipdb
        # ipdb.set_trace(context=20)
        loss_group = [self.compute_loss(s, t) for s, t in zip(s_fsp, t_fsp)]
        return loss_group

    @staticmethod
    def compute_loss(s, t):
        return (s - t).pow(2).mean()

    @staticmethod
    def compute_fsp(g):
        fsp_list = []

        for i in range(len(g) - 1):
            bot, top = g[i], g[i + 1]

            b_H, t_H = bot.shape[2], top.shape[2]
            if b_H > t_H:
                bot = F.adaptive_avg_pool2d(bot, (t_H, t_H))
            elif b_H < t_H:
                top = F.adaptive_avg_pool2d(top, (b_H, b_H))
            else:
                pass
            bot = bot.unsqueeze(1)
            top = top.unsqueeze(2)
            bot = bot.view(bot.shape[0], bot.shape[1], bot.shape[2], -1)
            top = top.view(top.shape[0], top.shape[1], top.shape[2], -1)

            fsp = (bot * top).mean(-1)
            fsp_list.append(fsp)
        return fsp_list

class RKDLoss(nn.Module):
    """Relational Knowledge Disitllation, CVPR2019"""
    def __init__(self, w_d=25, w_a=50):
        super(RKDLoss, self).__init__()
        self.w_d = w_d
        self.w_a = w_a

    def forward(self, f_s, f_t):
        student = f_s.view(f_s.shape[0], -1)
        teacher = f_t.view(f_t.shape[0], -1)

        # RKD distance loss
        with torch.no_grad():
            t_d = self.pdist(teacher, squared=False)
            mean_td = t_d[t_d > 0].mean()
            t_d = t_d / mean_td

        d = self.pdist(student, squared=False)
        mean_d = d[d > 0].mean()
        d = d / mean_d

        loss_d = F.smooth_l1_loss(d, t_d)

        # RKD Angle loss
        with torch.no_grad():
            td = (teacher.unsqueeze(0) - teacher.unsqueeze(1))
            norm_td = F.normalize(td, p=2, dim=2)
            t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)

        sd = (student.unsqueeze(0) - student.unsqueeze(1))
        norm_sd = F.normalize(sd, p=2, dim=2)
        s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)

        loss_a = F.smooth_l1_loss(s_angle, t_angle)

        loss = self.w_d * loss_d + self.w_a * loss_a

        return loss

    @staticmethod
    def pdist(e, squared=False, eps=1e-12):
        e_square = e.pow(2).sum(dim=1)
        prod = e @ e.t()
        res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

        if not squared:
            res = res.sqrt()

        res = res.clone()
        res[range(len(e)), range(len(e))] = 0
        return res