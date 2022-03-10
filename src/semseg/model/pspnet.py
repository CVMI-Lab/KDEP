import torch
from torch import nn
import torch.nn.functional as F

import model.resnet as models
import util.lovasz_softmax as L
import model.mobilev2 as mobilev2

class PPM(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins, BatchNorm):
        super(PPM, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                BatchNorm(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        return torch.cat(out, 1)


class PSPNet_TL(nn.Module):
    def __init__(self, layers=50, bins=(1, 2, 3, 6), dropout=0.1, classes=2, zoom_factor=8, use_ppm=True, criterion=nn.CrossEntropyLoss(ignore_index=255), BatchNorm=nn.BatchNorm2d, pretrained=True, initpath=None):
        super(PSPNet, self).__init__()
        assert layers in [18, 50, 101, 152]
        assert 2048 % len(bins) == 0
        assert classes > 1
        assert zoom_factor in [1, 2, 4, 8]
        self.zoom_factor = zoom_factor
        self.use_ppm = use_ppm
        self.criterion = criterion
        self.debug_gradient_norm = False
        self.layers = layers
        models.BatchNorm = BatchNorm

        if layers == 50:
            resnet = models.resnet50(pretrained=pretrained, deep_base=False, initpath=initpath) # deep_base=False for resnet50 from robust
        elif layers == 101:
            resnet = models.resnet101(pretrained=pretrained)
        elif layers == 18:
            resnet = models.resnet18(pretrained=pretrained, deep_base=False, initpath=initpath)
        else:
            resnet = models.resnet152(pretrained=pretrained)
        if layers == 18 or 50:
            self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
            fea_dim = 512
        else:
            self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.conv2, resnet.bn2, resnet.relu, resnet.conv3, resnet.bn3, resnet.relu, resnet.maxpool)
            fea_dim = 2048

        if layers == 50:
            fea_dim = 2048
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        if layers != 18:
            for n, m in self.layer3.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
            for n, m in self.layer4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
        else:
            for n, m in self.layer3.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                elif 'conv1' in n:
                    m.stride = (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
            for n, m in self.layer4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
                elif 'conv1' in n:
                    m.stride = (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)

        # fea_dim = 2048
        if use_ppm:
            self.ppm = PPM(fea_dim, int(fea_dim/len(bins)), bins, BatchNorm)
            fea_dim *= 2
        self.cls = nn.Sequential(
            nn.Conv2d(fea_dim, 512, kernel_size=3, padding=1, bias=False),
            BatchNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(512, classes, kernel_size=1)
        )
        if self.training:
            if layers != 18:
                self.aux = nn.Sequential(
                    nn.Conv2d(1024, 256, kernel_size=3, padding=1, bias=False),
                    BatchNorm(256),
                    nn.ReLU(inplace=True),
                    nn.Dropout2d(p=dropout),
                    nn.Conv2d(256, classes, kernel_size=1)
                )
            else:
                self.aux = nn.Sequential(
                    nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
                    BatchNorm(256),
                    nn.ReLU(inplace=True),
                    nn.Dropout2d(p=dropout),
                    nn.Conv2d(256, classes, kernel_size=1)
                )



    def criterion_lovasz(self,x,y,ignore=255):
        out = F.softmax(x, dim=1)
        loss = L.lovasz_softmax(out, y, ignore=ignore)
        # import ipdb
        # ipdb.set_trace(context=20)
        return loss

    def criterion_focal_loss(self,x,y,ignore=255):

        cross_entropy_loss = nn.CrossEntropyLoss(ignore_index=255,reduce=False)
        gamma = 2
        logpt = cross_entropy_loss(x,y)
        pt = torch.exp(-logpt)

        loss = ((1-pt)**gamma) * logpt
        loss = loss.mean()
        # import ipdb
        # ipdb.set_trace(context=20)
        return loss


    def forward(self, x, y=None, sup_loss_method='CE'):

        x_size = x.size()
        if self.layers not in [18, 50]:
            assert (x_size[2]-1) % 8 == 0 and (x_size[3]-1) % 8 == 0
        h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
        w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)

        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x_tmp = self.layer3(x)
        x = self.layer4(x_tmp)

        # import ipdb
        # ipdb.set_trace(context=20)

        if self.use_ppm:
            x = self.ppm(x)
        x = self.cls(x)
        if self.zoom_factor != 1:
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)

        if self.training:
            aux = self.aux(x_tmp)
            if self.zoom_factor != 1:
                aux = F.interpolate(aux, size=(h, w), mode='bilinear', align_corners=True)
            if y is not None:
                if sup_loss_method == 'CE':
                    main_loss = self.criterion(x, y)
                    aux_loss = self.criterion(aux, y)
                elif sup_loss_method == 'lovasz':
                    main_loss = self.criterion_lovasz(x, y)
                    aux_loss = self.criterion_lovasz(aux, y)
                elif sup_loss_method == 'MSE':
                    main_loss = self.criterion_sup_one_hot(x, y)
                    aux_loss = self.criterion_sup_one_hot(aux, y)
                elif sup_loss_method == 'focal':
                    main_loss = self.criterion_focal_loss(x, y)
                    aux_loss = self.criterion_focal_loss(aux, y)
            else:
                main_loss = None
                aux_loss = None
            if main_loss is not None:
                return (x, aux), main_loss, aux_loss
            else:
                return x

        else:
            aux = self.aux(x_tmp)
            if self.zoom_factor != 1:
                aux = F.interpolate(aux, size=(h, w), mode='bilinear', align_corners=True)

            if y is not None:
                main_loss = self.criterion(x, y)
                aux_loss = self.criterion(aux, y)
            else:
                main_loss = None
                aux_loss = None
            if main_loss is not None:
                return (x, aux), main_loss, aux_loss            # gradient norm
                # return x.max(1)[1], main_loss, aux_loss            # else
            else:
                return x

class PSPNet(nn.Module):
    def __init__(self, layers=50, bins=(1, 2, 3, 6), dropout=0.1, classes=2, zoom_factor=8, use_ppm=True, criterion=nn.CrossEntropyLoss(ignore_index=255), BatchNorm=nn.BatchNorm2d, pretrained=True, initpath=None):
        super(PSPNet, self).__init__()
        assert layers in [18, 50, 101, 152]
        assert 2048 % len(bins) == 0
        assert classes > 1
        assert zoom_factor in [1, 2, 4, 8]
        self.zoom_factor = zoom_factor
        self.use_ppm = use_ppm
        self.criterion = criterion
        self.debug_gradient_norm = False
        self.layers = layers
        models.BatchNorm = BatchNorm

        if layers == 50:
            resnet = models.resnet50(pretrained=pretrained, deep_base=False, initpath=initpath) # deep_base=False for resnet50 from robust
        elif layers == 101:
            resnet = models.resnet101(pretrained=pretrained)
        elif layers == 18:
            resnet = models.resnet18(pretrained=pretrained, deep_base=False, initpath=initpath)
        else:
            resnet = models.resnet152(pretrained=pretrained)
        if layers == 18 or layers == 50:
            self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
            fea_dim = 512
        else:
            self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.conv2, resnet.bn2, resnet.relu, resnet.conv3, resnet.bn3, resnet.relu, resnet.maxpool)
            fea_dim = 2048

        if layers == 50:
            fea_dim = 2048
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        if layers != 18:
            for n, m in self.layer3.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
            for n, m in self.layer4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
        else:
            for n, m in self.layer3.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                elif 'conv1' in n:
                    m.stride = (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
            for n, m in self.layer4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
                elif 'conv1' in n:
                    m.stride = (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)

        # fea_dim = 2048
        if use_ppm:
            self.ppm = PPM(fea_dim, int(fea_dim/len(bins)), bins, BatchNorm)
            fea_dim *= 2
        self.cls = nn.Sequential(
            nn.Conv2d(fea_dim, 512, kernel_size=3, padding=1, bias=False),
            BatchNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(512, classes, kernel_size=1)
        )
        if self.training:
            if layers != 18:
                self.aux = nn.Sequential(
                    nn.Conv2d(1024, 256, kernel_size=3, padding=1, bias=False),
                    BatchNorm(256),
                    nn.ReLU(inplace=True),
                    nn.Dropout2d(p=dropout),
                    nn.Conv2d(256, classes, kernel_size=1)
                )
            else:
                self.aux = nn.Sequential(
                    nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
                    BatchNorm(256),
                    nn.ReLU(inplace=True),
                    nn.Dropout2d(p=dropout),
                    nn.Conv2d(256, classes, kernel_size=1)
                )



    def criterion_lovasz(self,x,y,ignore=255):
        out = F.softmax(x, dim=1)
        loss = L.lovasz_softmax(out, y, ignore=ignore)
        # import ipdb
        # ipdb.set_trace(context=20)
        return loss

    def criterion_focal_loss(self,x,y,ignore=255):

        cross_entropy_loss = nn.CrossEntropyLoss(ignore_index=255,reduce=False)
        gamma = 2
        logpt = cross_entropy_loss(x,y)
        pt = torch.exp(-logpt)

        loss = ((1-pt)**gamma) * logpt
        loss = loss.mean()
        # import ipdb
        # ipdb.set_trace(context=20)
        return loss


    def forward(self, x, y=None, sup_loss_method='CE'):


        x_size = x.size()
        if self.layers not in [18, 50]:
            assert (x_size[2]-1) % 8 == 0 and (x_size[3]-1) % 8 == 0
        h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
        w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)

        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x_tmp = self.layer3(x)
        x = self.layer4(x_tmp)

        # import ipdb
        # ipdb.set_trace(context=20)

        if self.use_ppm:
            x = self.ppm(x)
        x = self.cls(x)
        if self.zoom_factor != 1:
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)

        if self.training:
            aux = self.aux(x_tmp)
            if self.zoom_factor != 1:
                aux = F.interpolate(aux, size=(h, w), mode='bilinear', align_corners=True)
            if y is not None:
                if sup_loss_method == 'CE':
                    main_loss = self.criterion(x, y)
                    aux_loss = self.criterion(aux, y)
                elif sup_loss_method == 'lovasz':
                    main_loss = self.criterion_lovasz(x, y)
                    aux_loss = self.criterion_lovasz(aux, y)
                elif sup_loss_method == 'MSE':
                    main_loss = self.criterion_sup_one_hot(x, y)
                    aux_loss = self.criterion_sup_one_hot(aux, y)
                elif sup_loss_method == 'focal':
                    main_loss = self.criterion_focal_loss(x, y)
                    aux_loss = self.criterion_focal_loss(aux, y)
            else:
                main_loss = None
                aux_loss = None
            if main_loss is not None:
                return (x, aux), main_loss, aux_loss
            else:
                return x

        else:
            aux = self.aux(x_tmp)
            if self.zoom_factor != 1:
                aux = F.interpolate(aux, size=(h, w), mode='bilinear', align_corners=True)

            if y is not None:
                main_loss = self.criterion(x, y)
                aux_loss = self.criterion(aux, y)
            else:
                main_loss = None
                aux_loss = None
            if main_loss is not None:
                return (x, aux), main_loss, aux_loss            # gradient norm
                # return x.max(1)[1], main_loss, aux_loss            # else
            else:
                return x

class Mobilev2_PSP(nn.Module):
    def __init__(self, layers=50, bins=(1, 2, 3, 6), dropout=0.1, classes=2, zoom_factor=8, use_ppm=True, criterion=nn.CrossEntropyLoss(ignore_index=255), BatchNorm=nn.BatchNorm2d, pretrained=True, initpath=None):
        super(Mobilev2_PSP, self).__init__()
        assert layers in [18, 50, 101, 152]
        assert 2048 % len(bins) == 0
        assert classes > 1
        assert zoom_factor in [1, 2, 4, 8]
        self.zoom_factor = zoom_factor
        self.use_ppm = use_ppm
        self.criterion = criterion
        self.debug_gradient_norm = False
        self.layers = layers
        models.BatchNorm = BatchNorm

        mobilenetv2=mobilev2.mobilenetv2(pretrained=pretrained, initpath=initpath)
        fea_dim = 1280
        self.features = mobilenetv2.features
        self.conv = mobilenetv2.conv

        if use_ppm:
            self.ppm = PPM(fea_dim, int(fea_dim/len(bins)), bins, BatchNorm)
            fea_dim *= 2
        self.cls = nn.Sequential(
            nn.Conv2d(fea_dim, 512, kernel_size=3, padding=1, bias=False),
            BatchNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(512, classes, kernel_size=1)
        )

    def criterion_lovasz(self,x,y,ignore=255):
        out = F.softmax(x, dim=1)
        loss = L.lovasz_softmax(out, y, ignore=ignore)
        # import ipdb
        # ipdb.set_trace(context=20)
        return loss

    def criterion_focal_loss(self,x,y,ignore=255):

        cross_entropy_loss = nn.CrossEntropyLoss(ignore_index=255,reduce=False)
        gamma = 2
        logpt = cross_entropy_loss(x,y)
        pt = torch.exp(-logpt)

        loss = ((1-pt)**gamma) * logpt
        loss = loss.mean()
        # import ipdb
        # ipdb.set_trace(context=20)
        return loss


    def forward(self, x, y=None, sup_loss_method='CE'):


        x_size = x.size()
        if self.layers not in [18, 50]:
            assert (x_size[2]-1) % 8 == 0 and (x_size[3]-1) % 8 == 0
        h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
        w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)

        x = self.features(x)
        # import ipdb
        # ipdb.set_trace(context=20)
        x = self.conv(x)

        if self.use_ppm:
            x = self.ppm(x)
        x = self.cls(x)
        if self.zoom_factor != 1:
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)

        if self.training:
            if y is not None:
                if sup_loss_method == 'CE':
                    main_loss = self.criterion(x, y)
                    aux_loss = main_loss
            else:
                main_loss = None
                aux_loss = None
            if main_loss is not None:
                return (x, x), main_loss, aux_loss
            else:
                return x

        else:
            if y is not None:
                main_loss = self.criterion(x, y)
                aux_loss = main_loss
            else:
                main_loss = None
                aux_loss = None
            if main_loss is not None:
                return (x, x), main_loss, aux_loss            # gradient norm
                # return x.max(1)[1], main_loss, aux_loss            # else
            else:
                return x


if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'
    input = torch.rand(4, 3, 512, 512).cuda()
    # model = PSPNet(layers=50, bins=(1, 2, 3, 6), dropout=0.1, classes=21, zoom_factor=1, use_ppm=True, pretrained=True).cuda()
    model = Mobilev2_PSP(layers=50, bins=(1, 2, 3, 6), dropout=0.1, classes=21, zoom_factor=1, use_ppm=True, pretrained=True).cuda()
    model.train()
    print(model)
    output = model(input)
    print('PSPNet', output.size())
