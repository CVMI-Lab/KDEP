import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
import model.transformer.net_factory as models
from model.transformer.prog_trans import PatchEmbed
import util.lovasz_softmax as L


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


class PSPNet(nn.Module):
    def __init__(self,
                 layers=50,
                 bins=(1, 2, 3, 6),
                 dropout=0.1,
                 classes=2,
                 zoom_factor=8,
                 use_ppm=True,
                 criterion=nn.CrossEntropyLoss(ignore_index=255),
                 BatchNorm=nn.BatchNorm2d,
                 input_size=672,
                 pretrained=True):
        super(PSPNet, self).__init__()
        assert layers in [50, 101, 152]
        assert 2048 % len(bins) == 0
        assert classes > 1
        assert zoom_factor in [1, 2, 4, 8]
        self.zoom_factor = zoom_factor
        self.use_ppm = use_ppm
        self.criterion = criterion
        self.debug_gradient_norm = False
        # models.BatchNorm = BatchNorm
        # net = models.prog_trans_small(pretrained=pretrained, BN=BatchNorm, input_size=input_size)
        net = models.prog_trans_small(pretrained=pretrained, model_path='./initmodel/model_best.pth.tar', BN=BatchNorm, input_size=input_size)
        self.rpn_tokens = net.rpn_tokens
        self.act = net.act
        self.conv = nn.Sequential(
            net.conv1,
            net.norm1,
            net.act,
            net.pool1
        )
        [setattr(self, f"layer{i}", getattr(net, f"layer_{i}")) for i in range(4)]
        self.last_linear = net.last_linear
        self.last_norm = net.last_norm

        for n, m in self.layer2.named_modules():
            if 'to_token' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
        for n, m in self.layer3.named_modules():
            if 'to_token' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)

        fea_dim = 768  # dim of the last layer of the network
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
            self.aux = nn.Sequential(
                nn.Conv2d(384, 256, kernel_size=3, padding=1, bias=False),
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
        _, _, input_h, input_w = x.size()
        # assert (x_size[2]-1) % 8 == 0 and (x_size[3]-1) % 8 == 0
        # h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
        # w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)

        rpn_tokens = self.rpn_tokens
        if rpn_tokens is not None:
            rpn_tokens = rpn_tokens.expand(x.shape[0], -1, -1)

        x = self.conv(x)
        b, c, h, w = x.shape
        x = rearrange(x, "b c h w -> b (h w) c")

        # if y is not None:
        #     mask = (y==255)
        #     mask = torch.unsqueeze(mask, dim=1)
        #     mask = mask.float()
        # else:
        #     mask = None

        mask=None

        x, h, w, _, rpn_tokens, mask = self.layer0(x, h, w, None, rpn_tokens, mask)
        x, h, w, _, rpn_tokens, mask = self.layer1(x, h, w, None, rpn_tokens, mask)
        x_tmp, h2, w2, _, rpn_tokens, mask = self.layer2(x, h, w, None, rpn_tokens, mask)
        x, h, w, _, rpn_tokens, mask = self.layer3(x_tmp, h2, w2, None, rpn_tokens, mask)
        # x = self.last_linear(x)
        # x = self.last_norm(x.permute(0, 2, 1)[..., None]).squeeze(-1)
        # x = self.act(x)
        x = x.permute(0, 2, 1)[..., None].squeeze(-1)
        x = rearrange(x, "b c (h w) -> b c h w", h=h)
        x_tmp = rearrange(x_tmp, "b (h w) c -> b c h w", h=h2)
        if self.use_ppm:
            x = self.ppm(x)
        x = self.cls(x)
        if self.zoom_factor != 1:
            x = F.interpolate(x, size=(input_h, input_w), mode='bilinear', align_corners=True)

        if self.training:
            aux = self.aux(x_tmp)
            if self.zoom_factor != 1:
                aux = F.interpolate(aux, size=(input_h, input_w), mode='bilinear', align_corners=True)
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
                aux = F.interpolate(aux, size=(input_h, input_w), mode='bilinear', align_corners=True)

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


if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'
    input = torch.rand(4, 3, 512, 672).cuda()
    model = PSPNet(layers=50, bins=(1, 2, 3, 6), dropout=0.1, classes=21, zoom_factor=1, use_ppm=True, pretrained=False).cuda()
    model.eval()
    print(model)
    output = model(input)
    print('PSPNet', output.size())
