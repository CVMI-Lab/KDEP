import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class Calib(nn.Module):
    '''
    Calib prediction (before softmax)
    For each class confidence value, update as: weight * z + bias
    '''
    def __init__(self, classes, desired_predict_confidence=0.9, weight=None, bias=None):
        super(Calib, self).__init__()
        self.classes = classes
        self.bias = nn.Parameter((torch.ones(classes) * 0.01).float())
        self.desired_predict_confidence = desired_predict_confidence
        if self.classes == 2:
            self.weight = nn.Parameter(torch.FloatTensor([1.1, 2.2]))

        if self.classes == 19:
            self.weight = nn.Parameter(
                0.5 * torch.FloatTensor([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]))
                # torch.FloatTensor([1.0101, 1.0204, 1.0101, 2.5000, 1.6667, 1.2500, 1.4286, 1.1111, 1.0101,
                #                    2.0000, 1.1111, 1.2500, 2.5000, 1.0101, 2.5000, 3.3333, 3.3333, 3.3333, 1.6667]))
            # self.weight = nn.Parameter(torch.FloatTensor([1.0101, 1.0204, 1.0101, 2.5000, 1.6667, 1.2500, 1.4286, 1.1111, 1.0101,
            #                             2.0000, 1.1111, 1.2500, 2.5000, 1.0101, 2.5000, 3.3333, 3.3333, 3.3333, 1.6667]))

    def forward(self, prediction, label=None):
        '''
        Calib
        :param prediction:  BCHW    B*19*1024*2048
        :param label: BHW
        :return:
        '''

        # weighted_prediction = self.weight.double() * prediction.double()
        # weighted_prediction += self.bias.double()
        # max_confidence = torch.max(weighted_prediction, dim=2)[0]
        # loss = 10000*torch.norm(max_confidence - self.desired_predict_confidence)/(1024*2048)

        prior_ratio = torch.FloatTensor([32.07,5.71,20.7, 0.564, 0.761,1.054, 0.1696, 0.5014,13.4993,0.8981,3.6445,1.1458, 0.1393,6.0, 0.2949, 0.1954, 0.2341, 0.0818, 0.3917])
        prior_ratio = prior_ratio / prior_ratio.sum()
        prior_ratio = prior_ratio.double()
        prior_ratio = prior_ratio.cuda()
        weighted_prediction = F.softmax(prediction, dim=1)
        weighted_prediction = self.weight.double() * weighted_prediction.double().permute(0,2,3,1)   # BHWC
        # weighted_prediction += self.bias.double()
        weighted_prediction = weighted_prediction.permute(0,3,1,2)      # -> BCHW

        norm_tensor = weighted_prediction.sum(1)
        weighted_prediction = weighted_prediction.permute(1,0,2,3)  # BCHW -> CBHW
        weighted_prediction = weighted_prediction * (1 / norm_tensor)
        weighted_prediction = weighted_prediction.permute(1,0,2,3)  # -> BCHW

        loss = 0
        threshold = 0.9

        size_all = torch.zeros(19).cuda()
        size_all = size_all.double()
        for j in range(19):
            # import ipdb
            # ipdb.set_trace(context=20)
            class_j_cf = weighted_prediction[:,j,:,:]   # BHW
            class_j_over_threshold = (class_j_cf > threshold)
            size_all[j] = class_j_over_threshold.sum()

        ratio_all = size_all / size_all.sum()
        loss = (ratio_all - prior_ratio)**2
        loss = loss.sum()

        # change label into one-hot
        ignore_area = (label == 255)

        one_hot = torch.zeros(label.shape[0], label.shape[1], label.shape[2], 19).cuda()  # BHWC
        one_hot = one_hot.double()
        for j in range(19):
            class_j_position = (label == j)
            vec = torch.zeros(19).cuda()
            vec[j] = 1
            vec = vec.double()
            one_hot[class_j_position] = vec
        one_hot = one_hot.permute(0, 3, 1, 2)  # BHWC -> BCHW

        # mse loss
        same_shape_as_input = F.mse_loss(weighted_prediction, one_hot, reduction='none')  # BCHW
        same_shape_as_input = same_shape_as_input.sum(1)  # BHW
        same_shape_as_input[ignore_area] = 0
        mse_loss = same_shape_as_input.mean()

        # print('distribute loss={}'.format(loss))
        # print('correctness loss={}'.format(mse_loss))
        all_loss = loss + 0.05 * mse_loss
        # all_loss = 0
        return all_loss, weighted_prediction
        # probas = F.softmax(weighted_prediction, dim=1)
        # criterion = nn.CrossEntropyLoss(ignore_index=255)
        # loss = criterion(weighted_prediction, label)
        # return loss

class Calib_distri(nn.Module):
    '''
    Calib prediction (before softmax)
    For each class confidence value, update as: softmax(weight * z + bias)
    '''
    def __init__(self, classes, init=0.5, weight=None, bias=None):
        super(Calib_distri, self).__init__()
        self.classes = classes
        self.bias = nn.Parameter((torch.ones(classes) * 0.01).float())
        # self.desired_predict_confidence = desired_predict_confidence
        if self.classes == 2:
            self.weight = nn.Parameter(torch.FloatTensor([1.1, 2.2]))

        if self.classes == 19:
            self.weight = nn.Parameter(
                init * torch.FloatTensor([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]))
                # torch.FloatTensor([1.0101, 1.0204, 1.0101, 2.5000, 1.6667, 1.2500, 1.4286, 1.1111, 1.0101,
                #                    2.0000, 1.1111, 1.2500, 2.5000, 1.0101, 2.5000, 3.3333, 3.3333, 3.3333, 1.6667]))
            # self.weight = nn.Parameter(torch.FloatTensor([1.0101, 1.0204, 1.0101, 2.5000, 1.6667, 1.2500, 1.4286, 1.1111, 1.0101,
            #                             2.0000, 1.1111, 1.2500, 2.5000, 1.0101, 2.5000, 3.3333, 3.3333, 3.3333, 1.6667]))

    def forward(self, prediction, label=None):
        '''
        Calib
        :param prediction:  BCHW    B*19*1024*2048
        :param label: BHW
        :return:
        '''

        weighted_prediction = self.weight.double() * prediction.double().permute(0,2,3,1)   # BHWC
        weighted_prediction += self.bias.double()
        scaled_logits = weighted_prediction.permute(0,3,1,2)      # -> BCHW


        # criterion = nn.CrossEntropyLoss(ignore_index=255)
        # loss = criterion(scaled_logits, label)
        scaled_logits = F.softmax(scaled_logits, dim=1)  # logits -> confidence

        cf_esti = scaled_logits.sum(2).sum(2) # BC
        norm_tensor = cf_esti.sum(1)[0]
        distri_esti = cf_esti / norm_tensor # BC
        distri_esti = distri_esti.float()

        if label is not None:
            K = self.classes
            distri_gt = torch.zeros(label.shape[0], K).cuda()
            for i in range(label.shape[0]):
                dis_label_i = torch.histc(label[i], bins=K, min=0, max=K - 1)
                dis_label_i = dis_label_i.float() / dis_label_i.sum(0)
                distri_gt[i] = dis_label_i
            # import ipdb
            # ipdb.set_trace(context=20)
            # loss = (distri_gt - distri_esti)**2
            # loss = loss.sum(0).sum(0)
            loss = torch.nn.functional.kl_div(distri_esti.log(), distri_gt)
            # print(loss)
            return loss, distri_esti
        else:
            return 0, distri_esti

class Calib_platt(nn.Module):
    '''
    Calib prediction (before softmax)
    For each class confidence value, update as: softmax(weight * z + bias)
    '''
    def __init__(self, classes, init=0.5, weight=None, bias=None):
        super(Calib_platt, self).__init__()
        self.classes = classes
        self.bias = nn.Parameter((torch.ones(classes) * 0.01).float())
        # self.desired_predict_confidence = desired_predict_confidence
        if self.classes == 2:
            self.weight = nn.Parameter(torch.FloatTensor([1.1, 2.2]))

        if self.classes == 19:
            self.weight = nn.Parameter(
                init * torch.FloatTensor([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]))
                # torch.FloatTensor([1.0101, 1.0204, 1.0101, 2.5000, 1.6667, 1.2500, 1.4286, 1.1111, 1.0101,
                #                    2.0000, 1.1111, 1.2500, 2.5000, 1.0101, 2.5000, 3.3333, 3.3333, 3.3333, 1.6667]))
            # self.weight = nn.Parameter(torch.FloatTensor([1.0101, 1.0204, 1.0101, 2.5000, 1.6667, 1.2500, 1.4286, 1.1111, 1.0101,
            #                             2.0000, 1.1111, 1.2500, 2.5000, 1.0101, 2.5000, 3.3333, 3.3333, 3.3333, 1.6667]))

    def forward(self, prediction, label=None):
        '''
        Calib
        :param prediction:  BCHW    B*19*1024*2048
        :param label: BHW
        :return:
        '''

        weighted_prediction = self.weight.double() * prediction.double().permute(0,2,3,1)   # BHWC
        weighted_prediction += self.bias.double()
        scaled_logits = weighted_prediction.permute(0,3,1,2)      # -> BCHW

        if self.training:
            criterion = nn.CrossEntropyLoss(ignore_index=255)
            loss = criterion(scaled_logits, label)
            # print(scaled_logits.shape)
            # import ipdb
            # ipdb.set_trace(context=20)
            weighted_prediction = F.softmax(scaled_logits, dim=1)
            return loss, weighted_prediction
        else:
            weighted_prediction = F.softmax(scaled_logits, dim=1)
            return 0, weighted_prediction


class Calib_T(nn.Module):
    '''
    Calib prediction (before softmax)
    prediction = softmax(logits / T)
    '''
    def __init__(self, init=1):
        super(Calib_T, self).__init__()
        self.T = nn.Parameter(init*(torch.ones(1)).double())

    def forward(self, prediction, label):
        '''
        Calib by T
        :param prediction: BCHW, logits
        :param label: BHW
        :return:
        '''

        scaled_logits = prediction.double() / self.T
        # probas = F.softmax(scaled_logits, dim=1)
        if self.training:
            criterion = nn.CrossEntropyLoss(ignore_index=255)
            loss = criterion(scaled_logits, label)
            # print(scaled_logits.shape)
            # import ipdb
            # ipdb.set_trace(context=20)
            weighted_prediction = F.softmax(scaled_logits, dim=1)
            return loss, weighted_prediction
        else:
            weighted_prediction = F.softmax(scaled_logits, dim=1)
            return 0, weighted_prediction

class embed_1x1(nn.Module):
    def __init__(self, output_channels):
        super(embed_1x1, self).__init__()
        self.theta = nn.Conv2d(in_channels=2048, out_channels=output_channels, kernel_size=1)
        self.phi = nn.Conv2d(in_channels=2048, out_channels=output_channels, kernel_size=1)

    def forward(self, feature, func='phi'):
        '''
        1x1 conv
        :param feature: BCHW
        :return:
        '''
        # import ipdb
        # ipdb.set_trace(context=20)
        if func == 'phi':
            ret = self.phi(feature)
        elif func == 'theta':
            ret = self.theta(feature)
        return ret


def hard_sigmoid(x):
    """
    Computes element-wise hard sigmoid of x.
    See e.g. https://github.com/Theano/Theano/blob/master/theano/tensor/nnet/sigm.py#L279
    """
    x = (0.2 * x) + 1
    x = F.threshold(-x, -2, -2)
    x = F.threshold(-x, 0, 0)
    return x

class Calib_Per_Sample_T(nn.Module):
    '''
    Use layers to predict T for each logits input
    '''
    def __init__(self, units=100, bs=2):
        super(Calib_Per_Sample_T, self).__init__()
        self.conv1 = nn.Conv2d(19, 1, kernel_size=3, stride=2,
                     padding=1, bias=False)
        self.conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=2,
                               padding=1, bias=False)
        self.avgpool = nn.AvgPool2d(7, stride=3)
        self.fc1 = nn.Linear(6888,units)
        self.fc2 = nn.Linear(units,bs)
        # self.sigmoid = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.T = torch.zeros(bs)

    def forward(self, prediction, label):
        '''
        Calib by T
        :param prediction: BCHW, logits
        :param label: BHW
        :return:
        '''

        x = prediction.float()
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, [2,2])
        x = self.conv2(x)
        x = F.relu(x)
        x = self.avgpool(x)
        x = x.flatten()

        x = self.fc1(x)
        x = self.fc2(x)
        x = 2 * self.sigmoid(x)
        # x = hard_sigmoid(x)
        # print(x)
        self.T = x
        # import ipdb
        # ipdb.set_trace(context=20)
        scaled_logits = prediction.permute(1,2,3,0) / self.T  # CHWB
        scaled_logits = scaled_logits.permute(3,0,1,2)        # BCHW
        criterion = nn.CrossEntropyLoss(ignore_index=255)
        # print(label.shape)
        # print(prediction.shape)
        loss = criterion(scaled_logits, label)
        return loss










