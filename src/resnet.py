import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import numpy as np
import copy

BatchNorm = nn.BatchNorm2d

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        # import ipdb
        # ipdb.set_trace(context=20)
        out += residual
        out = self.relu(out)

        return out

class BasicBlock_pre_relu(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, pre_relu=False):
        super(BasicBlock_pre_relu, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm(planes)
        self.downsample = downsample
        self.stride = stride
        self.pre_relu = pre_relu

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        # import ipdb
        # ipdb.set_trace(context=20)
        out += residual
        # out = self.relu(out)
        if self.pre_relu == False:
            return self.relu(out)
        else:
            # import ipdb
            # ipdb.set_trace(context=20)
            return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = BatchNorm(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck_pre_relu(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, pre_relu=False):
        super(Bottleneck_pre_relu, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = BatchNorm(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.pre_relu = pre_relu

    def forward(self, x):
        residual = x

        # import ipdb
        # ipdb.set_trace(context=20)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        # pre = copy.deepcopy(out)
        # out = self.relu(out)
        if self.pre_relu == False:
            return self.relu(out)
        else:
            # import ipdb
            # ipdb.set_trace(context=20)
            return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, deep_base=True):
        super(ResNet, self).__init__()
        self.deep_base = deep_base
        if not self.deep_base:
            self.inplanes = 64
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = BatchNorm(64)
        else:
            self.inplanes = 128
            self.conv1 = conv3x3(3, 64, stride=2)
            self.bn1 = BatchNorm(64)
            self.conv2 = conv3x3(64, 64)
            self.bn2 = BatchNorm(64)
            self.conv3 = conv3x3(64, 128)
            self.bn3 = BatchNorm(128)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, BatchNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, all_feat=False):
        x = self.relu(self.bn1(self.conv1(x)))
        if self.deep_base:
            x = self.relu(self.bn2(self.conv2(x)))
            x = self.relu(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # import ipdb
        # ipdb.set_trace(context=20)

        x = self.avgpool(x)
        feat = x.squeeze()  # B 2048
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x, feat


class ResNet_feat(nn.Module):

    def __init__(self, block, layers, num_classes=1000, deep_base=True, emb_dim=2048):
        super(ResNet_feat, self).__init__()
        self.deep_base = deep_base
        if not self.deep_base:
            self.inplanes = 64
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = BatchNorm(64)
        else:
            self.inplanes = 128
            self.conv1 = conv3x3(3, 64, stride=2)
            self.bn1 = BatchNorm(64)
            self.conv2 = conv3x3(64, 64)
            self.bn2 = BatchNorm(64)
            self.conv3 = conv3x3(64, 128)
            self.bn3 = BatchNorm(128)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.emb = nn.Linear(512 * block.expansion, emb_dim)
        self.l2norm = nn.BatchNorm1d(emb_dim)
        # self.emb = nn.AdaptiveAvgPool1d(emb_dim)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, BatchNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        nn.init.kaiming_normal_(self.emb.weight, mode='fan_out')
        nn.init.constant_(self.emb.bias, 0)
        nn.init.constant_(self.l2norm.weight, 1)
        nn.init.constant_(self.l2norm.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, all_feat=False):
        x = self.relu(self.bn1(self.conv1(x)))
        if self.deep_base:
            x = self.relu(self.bn2(self.conv2(x)))
            x = self.relu(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        f0 = x

        x = self.layer1(x)
        f1 = x
        x = self.layer2(x)
        f2 = x
        x = self.layer3(x)
        f3 = x
        x = self.layer4(x)
        f4 = x


        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        feat = self.emb(x) # B 2048
        feat = self.l2norm(feat)

        # import ipdb
        # ipdb.set_trace(context=20)

        # feat = self.relu(feat)
        x = self.fc(x)

        if all_feat is False:
            return x,feat
        else:
            return x, [f0,f1,f2,f3,f4]

class ResNet_feat_pre_relu(nn.Module):

    def __init__(self, block, layers, num_classes=1000, deep_base=True, emb_dim=2048):
        super(ResNet_feat_pre_relu, self).__init__()
        self.deep_base = deep_base
        if not self.deep_base:
            self.inplanes = 64
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = BatchNorm(64)
        else:
            self.inplanes = 128
            self.conv1 = conv3x3(3, 64, stride=2)
            self.bn1 = BatchNorm(64)
            self.conv2 = conv3x3(64, 64)
            self.bn2 = BatchNorm(64)
            self.conv3 = conv3x3(64, 128)
            self.bn3 = BatchNorm(128)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, last_pre_relu=True)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # self.feat_scale = nn.Parameter(torch.ones([]) * 3.0)
        # self.feat_scale = nn.Parameter(torch.ones([]) * np.log(1.0))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, BatchNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, last_pre_relu=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        if last_pre_relu == False:
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes))
        else:
            for i in range(1, blocks - 1):
                layers.append(block(self.inplanes, planes))
            layers.append(block(self.inplanes, planes, pre_relu=True))

        return nn.Sequential(*layers)

    def get_bn_before_relu(self):
        if isinstance(self.layer1[0], Bottleneck_pre_relu):
            bn4 = self.layer4[-1].bn3
        elif isinstance(self.layer1[0], BasicBlock_pre_relu):
            bn4 = self.layer4[-1].bn2
        else:
            print('ResNet unknown block error !!!')

        return bn4

    def forward(self, x, all_feat=False):
        # with torch.no_grad():
        x = self.relu(self.bn1(self.conv1(x)))
        if self.deep_base:
            x = self.relu(self.bn2(self.conv2(x)))
            x = self.relu(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        f0 = x

        x = self.layer1(x)
        f1 = x
        x = self.layer2(x)
        f2 = x
        x = self.layer3(x)
        f3 = x
        x = self.layer4(x) # pre relu
        f4 = x

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        feat = x
        # feat = feat * 3
        # feat = x * self.feat_scale.exp() # add feat scale
        x = self.fc(x)
        # import ipdb
        # ipdb.set_trace(context=20)

        if all_feat is False:
            return x,feat
        else:
            return x, [f0,f1,f2,f3,f4]

class ResNet_feat_svd_pre_relu(nn.Module):

    def __init__(self, block, layers, num_classes=1000, deep_base=True, emb_dim=2048):
        super(ResNet_feat_svd_pre_relu, self).__init__()
        self.deep_base = deep_base
        if not self.deep_base:
            self.inplanes = 64
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = BatchNorm(64)
        else:
            self.inplanes = 128
            self.conv1 = conv3x3(3, 64, stride=2)
            self.bn1 = BatchNorm(64)
            self.conv2 = conv3x3(64, 64)
            self.bn2 = BatchNorm(64)
            self.conv3 = conv3x3(64, 128)
            self.bn3 = BatchNorm(128)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, last_pre_relu=True)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        # self.interpolate = nn.functional.interpolate(size=512)
        # self.l2norm = nn.BatchNorm1d(emb_dim)
        # self.emb = nn.AdaptiveAvgPool1d(emb_dim)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # ---- res50 svd 512d
        self.svd_v = np.load('pretrained-models/StandardSP-ImageNet1K-ResNet50_v.npy')
        self.svd_v = np.load('pretrained-models/Microsoft-ResNet50_v.npy')

        self.svd_mean = np.load('pretrained-models/StandardSP-ImageNet1K-ResNet50_feat_mean.npy')
        self.svd_mean = np.load('pretrained-models/Microsoft-ResNet50_feat_mean.npy')

        self.svd_v = torch.from_numpy(self.svd_v)
        self.svd_mean = torch.from_numpy(self.svd_mean)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, BatchNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # nn.init.kaiming_normal_(self.emb.weight, mode='fan_out')
        # nn.init.constant_(self.emb.bias, 0)
        # nn.init.constant_(self.l2norm.weight, 1)
        # nn.init.constant_(self.l2norm.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1,last_pre_relu=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        if last_pre_relu==False:
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes))
        else:
            for i in range(1, blocks-1):
                layers.append(block(self.inplanes, planes))
            layers.append(block(self.inplanes, planes, pre_relu=True))

        return nn.Sequential(*layers)

    def forward(self, x, all_feat=False):
        with torch.no_grad():
            x = self.relu(self.bn1(self.conv1(x)))
            if self.deep_base:
                x = self.relu(self.bn2(self.conv2(x)))
                x = self.relu(self.bn3(self.conv3(x)))
            x = self.maxpool(x)
            f0 = x

            x = self.layer1(x)
            f1 = x
            x = self.layer2(x)
            f2 = x
            x = self.layer3(x)
            f3 = x
            x = self.layer4(x) # pre_relu
            f4 = x

            x = self.avgpool(x)

            x = x.view(x.size(0), -1)     # B 2048

            feat=x

            # --- directly select by var
            # import ipdb
            # ipdb.set_trace(context=20)
            # feat = x
            # # feat = torch.index_select(feat, 1, self.var_indice[:512].cuda())
            # feat = torch.index_select(feat, 1, self.var_indice[:1280].cuda())

            # --- interpolate
            # feat = x.view(x.size(0), 1, -1)  # B 1 2048
            # # feat = nn.functional.interpolate(feat, size=512)  # B 1 512
            # feat = nn.functional.interpolate(feat, size=1280)  # B 1 512
            # feat = feat.view(feat.size(0), -1)  # B 512

            # ----- random select 512 from original 2048 feat
            # feat = x
            # feat = feat[:,rand_indice_1280]
            # import ipdb
            # ipdb.set_trace(context=20)

            # ----- PCA
            feat = x - self.svd_mean.cuda()
            feat = torch.matmul(feat, self.svd_v[:, :512].cuda())  # B 512
            # # feat = torch.matmul(feat, self.svd_v[:, :1280].cuda())  # B 1280


            # # # -------- PTS Function Processing
            feat = torch.sign(feat) * torch.pow(torch.abs(feat / 0.1), 1 / 3)
            # feat = torch.sign(feat) * torch.pow(torch.abs(feat / 0.03), 1 / 3)
            # feat = torch.sign(feat) * torch.pow(torch.abs(feat / 0.2), 1 / 2)

            # ----------- Var matching
            # ---- step1:  all feat channels to have the same variance
            # svd_eigen = self.svd_eigen[:512].cuda()
            # # svd_eigen = self.svd_eigen[:1280].cuda()
            # # eigen_div_term = 1.0 / torch.sqrt(svd_eigen)
            # eigen_div_term = 1.0 / svd_eigen
            #
            # feat = feat*eigen_div_term
            # feat = feat * 500

            # ---- step2: variance matching with pre-svd feature
            # -- step2 > method 1: (Max-x)/(x-Min) == (STD-y)/(y-1); y=(Max-STD*Min+(STD-1)*x) / (Max-Min)
            # STD_PRE_PCA = 2.98
            # # STD_PRE_PCA = 4.57
            # matching_term = (svd_eigen[0]-STD_PRE_PCA*svd_eigen[-1]+(STD_PRE_PCA-1)*svd_eigen) / (svd_eigen[0]-svd_eigen[-1]) # fix bug
            # matching_term[0] = STD_PRE_PCA
            # matching_term[-1] = 1
            #
            # feat = feat * matching_term
            # # import ipdb
            # # ipdb.set_trace(context=20)
            # feat = feat * 357
            # # feat = feat * 347
            # -- step2 > method 1 --------------------------------------------------------------------------------------------------------------

            # -- step2 > method 2: match to orig pre relu variance
            # sorted_orig_var, _ = torch.sort(self.orig_var.cuda(), descending=True)
            # matching_term = sorted_orig_var[:512]
            # # matching_term = sorted_orig_var[:1280]
            # feat = feat * matching_term
            # feat = feat * 357
            # -- step2 > method 2 --------------------------------------------------------------------------------------------------------------



            # feat = feat*eigen_div_term * (1.0/eigen_div_term.mean())
            # feat = feat*eigen_div_term * (1.0/eigen_div_term.min())
            # feat = feat*eigen_div_term * (1.0/eigen_div_term.max())

            # feat = torch.matmul(feat, self.svd_v[:, -512:].cuda())  # B smallest eigen values 512
            # feat = torch.matmul(feat, self.svd_v[:, 512:1024].cuda())  # B smallest eigen values 512
            # feat = torch.matmul(feat, self.svd_v[:, rand_indice].cuda())  # B random indice 512

            # feat = self.relu(feat) # if apply relu after svd

            x = self.fc(x)

            if all_feat is False:
                return x,feat
            else:
                return x, [f0,f1,f2,f3,f4]


def resnet18(pretrained=False, initpath=None, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], deep_base=False, **kwargs)
    if pretrained and initpath is not None:
        # model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
        # model_path = './initmodel/transformedresnet-18-l2-eps0.ckpt'
        model_path = initpath
        model.load_state_dict(torch.load(model_path), strict=False)
        print('loaded from %s'%initpath)
    return model


def resnet18_feat(pretrained=False, initpath=None, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_feat(BasicBlock, [2, 2, 2, 2], deep_base=False, **kwargs)
    if pretrained and initpath is not None:
        # model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
        # model_path = './initmodel/transformedresnet-18-l2-eps0.ckpt'
        model_path = initpath
        model.load_state_dict(torch.load(model_path), strict=False)
        print('loaded from %s'%initpath)
    return model


def resnet18_feat_pre_relu(pretrained=False, initpath=None, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_feat_pre_relu(BasicBlock_pre_relu, [2, 2, 2, 2], deep_base=False, **kwargs)
    if pretrained and initpath is not None:
        # model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
        # model_path = './initmodel/transformedresnet-18-l2-eps0.ckpt'
        model_path = initpath
        model.load_state_dict(torch.load(model_path), strict=False)
        print('loaded from %s'%initpath)
    return model


def resnet34(pretrained=False, initpath=None, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3],deep_base=False, **kwargs)
    if pretrained and initpath is not None:
        # model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
        # model_path = './initmodel/transformedresnet-18-l2-eps0.ckpt'
        model_path = initpath
        model.load_state_dict(torch.load(model_path), strict=False)
        print('loaded from %s' % initpath)
    return model


def resnet50(pretrained=False, initpath=None, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    # model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    model = ResNet(Bottleneck, [3, 4, 6, 3], deep_base=False, **kwargs)
    if pretrained and initpath is not None:
        # model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
        # model_path = './initmodel/resnet50_v2.pth'
        model_path = initpath
        # model_path = '/root/Workspace_he/from-git/HKU/PspNet/SSL-PSP/initmodel/resnet50_v2.pth'
        model.load_state_dict(torch.load(model_path), strict=True)
        # print('loaded from initmodel/resnet50_v2.pth')
        print('loaded from %s' % initpath)
    return model

def resnet50_feat(pretrained=False, initpath=None, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_feat(Bottleneck, [3, 4, 6, 3], deep_base=False, **kwargs)
    if pretrained and initpath is not None:
        # model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
        # model_path = './initmodel/resnet50_v2.pth'
        model_path = initpath
        # model_path = '/root/Workspace_he/from-git/HKU/PspNet/SSL-PSP/initmodel/resnet50_v2.pth'
        model.load_state_dict(torch.load(model_path), strict=False)
        # print('loaded from initmodel/resnet50_v2.pth')
        print('loaded from %s' % initpath)
    return model


def resnet50_feat_svd_pre_relu(pretrained=False, initpath=None, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_feat_svd_pre_relu(Bottleneck_pre_relu, [3, 4, 6, 3], deep_base=False, **kwargs)
    if pretrained and initpath is not None:
        # model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
        # model_path = './initmodel/resnet50_v2.pth'
        model_path = initpath
        # model_path = '/root/Workspace_he/from-git/HKU/PspNet/SSL-PSP/initmodel/resnet50_v2.pth'
        model.load_state_dict(torch.load(model_path), strict=False)
        # print('loaded from initmodel/resnet50_v2.pth')
        print('loaded from %s' % initpath)
    return model

def resnet50_feat_pre_relu(pretrained=False, initpath=None, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_feat_pre_relu(Bottleneck_pre_relu, [3, 4, 6, 3], deep_base=False, **kwargs)
    if pretrained and initpath is not None:
        # model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
        # model_path = './initmodel/resnet50_v2.pth'
        model_path = initpath
        # model_path = '/root/Workspace_he/from-git/HKU/PspNet/SSL-PSP/initmodel/resnet50_v2.pth'
        model.load_state_dict(torch.load(model_path), strict=False)
        # print('loaded from initmodel/resnet50_v2.pth')
        print('loaded from %s' % initpath)
    return model

def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        # model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
        model_path = './initmodel/resnet101_v2.pth'
        model.load_state_dict(torch.load(model_path), strict=False)
    return model


def resnet152(pretrained=False, initpath=None, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], deep_base=False, **kwargs)
    if pretrained and initpath is not None:
        # model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
        # model_path = './initmodel/resnet50_v2.pth'
        model_path = initpath
        # model_path = '/root/Workspace_he/from-git/HKU/PspNet/SSL-PSP/initmodel/resnet50_v2.pth'
        model.load_state_dict(torch.load(model_path), strict=False)
        # print('loaded from initmodel/resnet50_v2.pth')
        print('loaded from %s' % initpath)
    return model
