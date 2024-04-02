import torch
import torch.nn as nn
import sys
import math

sys.path.append('../../')

from lightconvpoint.nn import Convolution_FKAConv as Conv
from lightconvpoint.nn import max_pool, interpolate
from lightconvpoint.spatial import knn, sampling_quantized as sampling

NormLayer = nn.BatchNorm1d


class SA_Layer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x_q = self.q_conv(x).permute(0, 2, 1)  # b, n, c
        x_k = self.k_conv(x)  # b, c, n
        x_v = self.v_conv(x)
        energy = x_q @ x_k  # b, n, n
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))
        x_r = x_v @ attention  # b, c, n
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x


class GCA(nn.Module):
    def __init__(self, channels_high, channels_low):
        super(GCA, self).__init__()
        # Global Context Attention
        self.conv3x3 = nn.Conv1d(channels_low, channels_low, kernel_size=3, padding=1, bias=False)
        self.bn_low = nn.BatchNorm1d(channels_low)
        self.conv1x1 = nn.Conv1d(channels_high, channels_low, kernel_size=1, padding=0, bias=False)
        self.bn_high = nn.BatchNorm1d(channels_low)
        self.conv_reduction = nn.Conv1d(channels_low, channels_high, kernel_size=1, padding=0, bias=False)
        self.bn_reduction = nn.BatchNorm1d(channels_high)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, fms_high, fms_low):
        b, c, h = fms_high.shape

        fms_high_gp = nn.AvgPool1d(fms_high.shape[2:])(fms_high).view(len(fms_high), c, 1)

        fms_high_gp = self.conv1x1(fms_high_gp)
        fms_high_gp = self.bn_high(fms_high_gp)
        fms_high_gp = self.relu(fms_high_gp)

        fms_low = nn.AdaptiveAvgPool1d(fms_high.shape[2:])(fms_low)
        fms_low_mask = self.conv3x3(fms_low)
        fms_low_mask = self.bn_low(fms_low_mask)
        fms_att = fms_low_mask * fms_high_gp

        out = self.relu(self.bn_reduction(self.conv_reduction(fms_att)))
        return out


class ECA_block(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(ECA_block, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.transpose(-1, -2)).transpose(-1, -2)
        y = self.sigmoid(y)
        out = x * y.expand_as(x)
        return out


class ResidualBlock_Base(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, adaptive_normalization=True):
        super().__init__()

        self.cv0 = nn.Conv1d(in_channels, in_channels // 2, 1)
        self.bn0 = NormLayer(in_channels // 2)
        self.cv1 = Conv(in_channels // 2, in_channels // 2, kernel_size, adaptive_normalization=adaptive_normalization)
        self.bn1 = NormLayer(in_channels // 2)
        self.cv2 = nn.Conv1d(in_channels // 2, out_channels, 1)
        self.bn2 = NormLayer(out_channels)
        self.activation = nn.ReLU(inplace=True)

        self.shortcut = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.bn_shortcut = NormLayer(out_channels) if in_channels != out_channels else nn.Identity()

    def forward(self, x, pos, support_points, neighbors_indices):
        x_short = x
        x = self.activation(self.bn0(self.cv0(x)))
        x = self.activation(self.bn1(self.cv1(x, pos, support_points, neighbors_indices)))
        x = self.bn2(self.cv2(x))

        x_short = self.bn_shortcut(self.shortcut(x_short))
        if x_short.shape[2] != x.shape[2]:
            x_short = max_pool(x_short, neighbors_indices)

        x = self.activation(x + x_short)

        return x

class ResidualBlock(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, adaptive_normalization=True):
        super().__init__()

        self.cv0 = nn.Conv1d(in_channels, in_channels // 2, 1)
        self.bn0 = NormLayer(in_channels // 2)
        self.cv1 = Conv(in_channels // 2, in_channels // 2, kernel_size, adaptive_normalization=adaptive_normalization)
        self.bn1 = NormLayer(in_channels // 2)
        self.cv2 = nn.Conv1d(in_channels // 2, out_channels, 1)
        self.bn2 = NormLayer(out_channels)
        self.activation = nn.ReLU(inplace=True)


        self.shortcut = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.bn_shortcut = NormLayer(out_channels) if in_channels != out_channels else nn.Identity()
        self.eca = ECA_block(out_channels)
        self.eca1 = ECA_block(out_channels)

        self.sa_para1 = nn.Parameter(torch.ones(2), requires_grad=True)
        self.sa_para2 = nn.Parameter(torch.ones(2), requires_grad=True)

    def forward(self, x, pos, support_points, neighbors_indices):
        x_short = x
        x = self.activation(self.bn0(self.cv0(x)))
        x = self.activation(self.bn1(self.cv1(x, pos, support_points, neighbors_indices)))
        x = self.bn2(self.cv2(x))

        x_short = self.bn_shortcut(self.shortcut(x_short))
        if x_short.shape[2] != x.shape[2]:
            x_short = max_pool(x_short, neighbors_indices)
        x_short1=self.eca(x_short)

        x_abs = torch.abs(x)
        thres = self.eca1(x_abs)

        sub = x_abs - thres
        zeros = thres - thres
        n_sub = torch.maximum(sub, zeros)+1

        x1 = x * n_sub

        x = self.activation(self.sa_para1[0]*x+self.sa_para1[1]*x1 + self.sa_para2[0]*x_short+self.sa_para2[1]*x_short1)
        return x


class FKAConvNetwork(torch.nn.Module):

    def __init__(self, in_channels, out_channels, segmentation=False, hidden=64, dropout=0.5,
                 last_layer_additional_size=None, adaptive_normalization=True, fix_support_number=False):
        super().__init__()

        self.lcp_preprocess = True
        self.segmentation = segmentation
        self.adaptive_normalization = adaptive_normalization
        self.fix_support_point_number = fix_support_number

        self.cv0 = Conv(in_channels, hidden, 16, adaptive_normalization=self.adaptive_normalization)
        self.bn0 = NormLayer(hidden)

        self.resnetb01 = ResidualBlock_Base(hidden, hidden, 16, adaptive_normalization=self.adaptive_normalization)

        self.resnetb10 = ResidualBlock_Base(hidden, 2 * hidden, 16, adaptive_normalization=self.adaptive_normalization)
        self.resnetb11 = ResidualBlock(2 * hidden, 2 * hidden, 16, adaptive_normalization=self.adaptive_normalization)
        self.sa_1 = SA_Layer(2 * hidden)
        self.sa_para1_1 = nn.Parameter(torch.ones(2), requires_grad=True)
        self.sa_para1_2 = nn.Parameter(torch.ones(2), requires_grad=True)
        self.gca_1 = GCA(2 * hidden, hidden)

        self.resnetb20 = ResidualBlock_Base(2 * hidden, 4 * hidden, 16, adaptive_normalization=self.adaptive_normalization)
        self.resnetb21 = ResidualBlock(4 * hidden, 4 * hidden, 16, adaptive_normalization=self.adaptive_normalization)
        self.sa_2 = SA_Layer(4 * hidden)
        self.sa_para2_1 = nn.Parameter(torch.ones(2), requires_grad=True)
        self.sa_para2_2 = nn.Parameter(torch.ones(2), requires_grad=True)
        self.gca_2 = GCA(4 * hidden, 2*hidden)


        self.resnetb30 = ResidualBlock_Base(4 * hidden, 8 * hidden, 16, adaptive_normalization=self.adaptive_normalization)
        self.resnetb31 = ResidualBlock(8 * hidden, 8 * hidden, 16, adaptive_normalization=self.adaptive_normalization)
        self.sa_3 = SA_Layer(8 * hidden)
        self.sa_para3_1 = nn.Parameter(torch.ones(2), requires_grad=True)
        self.sa_para3_2 = nn.Parameter(torch.ones(2), requires_grad=True)
        self.gca_3 = GCA(8 * hidden, 4 * hidden)

        self.resnetb40 = ResidualBlock_Base(8 * hidden, 16 * hidden, 16, adaptive_normalization=self.adaptive_normalization)
        self.resnetb41 = ResidualBlock(16 * hidden, 16 * hidden, 16, adaptive_normalization=self.adaptive_normalization)
        self.sa_4 = SA_Layer(16 * hidden)
        self.sa_para4_1 = nn.Parameter(torch.ones(2), requires_grad=True)
        self.sa_para4_2 = nn.Parameter(torch.ones(2), requires_grad=True)
        self.gca_4 = GCA(16 * hidden, 8 * hidden)

        if self.segmentation:
            self.cv5 = nn.Conv1d(32 * hidden, 16 * hidden, 1)
            self.bn5 = NormLayer(16 * hidden)
            self.cv3d = nn.Conv1d(24 * hidden, 8 * hidden, 1)
            self.bn3d = NormLayer(8 * hidden)
            self.cv2d = nn.Conv1d(12 * hidden, 4 * hidden, 1)
            self.bn2d = NormLayer(4 * hidden)
            self.cv1d = nn.Conv1d(6 * hidden, 2 * hidden, 1)
            self.bn1d = NormLayer(2 * hidden)
            self.cv0d = nn.Conv1d(3 * hidden, hidden, 1)
            self.bn0d = NormLayer(hidden)

            if last_layer_additional_size is not None:
                self.fcout = nn.Conv1d(hidden + last_layer_additional_size, out_channels, 1)
            else:
                self.fcout = nn.Conv1d(hidden, out_channels, 1)
        else:
            self.fcout = nn.Conv1d(16 * hidden, out_channels, 1)

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward_spatial(self, data):

        pos = data["pos"].clone()

        add_batch_dimension = False
        if len(pos.shape) == 2:
            pos = pos.unsqueeze(0)
            add_batch_dimension = True

        # compute the support points
        if self.fix_support_point_number:
            support1, _ = sampling(pos, n_support=512)
            support2, _ = sampling(support1, n_support=128)
            support3, _ = sampling(support2, n_support=32)
            support4, _ = sampling(support3, n_support=8)
        else:
            support1, _ = sampling(pos, 0.25)
            support2, _ = sampling(support1, 0.25)
            support3, _ = sampling(support2, 0.25)
            support4, _ = sampling(support3, 0.25)

        # compute the ids
        ids00 = knn(pos, pos, 16)
        ids01 = knn(pos, support1, 16)
        ids11 = knn(support1, support1, 16)
        ids12 = knn(support1, support2, 16)
        ids22 = knn(support2, support2, 16)
        ids23 = knn(support2, support3, 16)
        ids33 = knn(support3, support3, 16)
        ids34 = knn(support3, support4, 16)
        ids44 = knn(support4, support4, 16)
        if self.segmentation:
            ids43 = knn(support4, support3, 1)
            ids32 = knn(support3, support2, 1)
            ids21 = knn(support2, support1, 1)
            ids10 = knn(support1, pos, 1)

        ret_data = {}
        if add_batch_dimension:
            support1 = support1.squeeze(0)
            support2 = support2.squeeze(0)
            support3 = support3.squeeze(0)
            support4 = support4.squeeze(0)
            ids00 = ids00.squeeze(0)
            ids01 = ids01.squeeze(0)
            ids11 = ids11.squeeze(0)
            ids12 = ids12.squeeze(0)
            ids22 = ids22.squeeze(0)
            ids23 = ids23.squeeze(0)
            ids33 = ids33.squeeze(0)
            ids34 = ids34.squeeze(0)
            ids44 = ids44.squeeze(0)

        ret_data["support1"] = support1
        ret_data["support2"] = support2
        ret_data["support3"] = support3
        ret_data["support4"] = support4

        ret_data["ids00"] = ids00
        ret_data["ids01"] = ids01
        ret_data["ids11"] = ids11
        ret_data["ids12"] = ids12
        ret_data["ids22"] = ids22
        ret_data["ids23"] = ids23
        ret_data["ids33"] = ids33
        ret_data["ids34"] = ids34
        ret_data["ids44"] = ids44

        if self.segmentation:
            if add_batch_dimension:
                ids43 = ids43.squeeze(0)
                ids32 = ids32.squeeze(0)
                ids21 = ids21.squeeze(0)
                ids10 = ids10.squeeze(0)

            ret_data["ids43"] = ids43
            ret_data["ids32"] = ids32
            ret_data["ids21"] = ids21
            ret_data["ids10"] = ids10

        return ret_data

    def forward(self, data, spatial_only=False, spectral_only=False, cat_in_last_layer=None):

        if spatial_only:
            return self.forward_spatial(data)

        if not spectral_only:

            spatial_data = self.forward_spatial(data)
            for key, value in spatial_data.items():
                data[key] = value
            # data = {**data, **spatial_data}

        x = data["x"]
        # batch_size*3*3000
        pos = data["pos"]
        # batch_size*3*3000

        x0 = self.activation(self.bn0(self.cv0(x, pos, pos, data["ids00"])))
        x0 = self.resnetb01(x0, pos, pos, data["ids00"])
        # batch_size*64*3000
        x1 = self.resnetb10(x0, pos, data["support1"], data["ids01"])
        x1 = self.resnetb11(x1, data["support1"], data["support1"], data["ids11"])
        x1 = self.sa_para1_1[0] * x1 + self.sa_para1_1[1] * self.sa_1(x1)
        x01 =  self.sa_para1_2[0] * x1+self.sa_para1_2[1] * self.gca_1(x1,x0)

        # batch_size*128*750
        x2 = self.resnetb20(x01, data["support1"], data["support2"], data["ids12"])
        x2 = self.resnetb21(x2, data["support2"], data["support2"], data["ids22"])
        x2 = self.sa_para2_1[0] * x2 + self.sa_para2_1[1] * self.sa_2(x2)
        x12 =  self.sa_para2_2[0] * x2+self.sa_para2_2[1] * self.gca_2(x2, x1)

        # batch_size*256*187
        x3 = self.resnetb30(x12, data["support2"], data["support3"], data["ids23"])
        x3 = self.resnetb31(x3, data["support3"], data["support3"], data["ids33"])
        x3 = self.sa_para3_1[0] * x3 + self.sa_para3_1[1] * self.sa_3(x3)
        x23 =  self.sa_para3_2[0] * x3+self.sa_para3_2[1] * self.gca_3(x3, x2)

        # batch_size*512*46
        x4 = self.resnetb40(x23, data["support3"], data["support4"], data["ids34"])
        x4 = self.resnetb41(x4, data["support4"], data["support4"], data["ids44"])
        x4 = self.sa_para4_1[0] * x4 + self.sa_para4_1[1] * self.sa_4(x4)
        x4 = self.sa_para4_2[0] * x4+self.sa_para4_2[1] * self.gca_4(x4, x3)
        # batch_size*1024*11

        if self.segmentation:

            x5 = x4.max(dim=2, keepdim=True)[0].expand_as(x4)
            x4d = self.activation(self.bn5(self.cv5(torch.cat([x4, x5], dim=1))))
            x4d = x4

            x3d = interpolate(x4d, data["ids43"])
            x3d = self.activation(self.bn3d(self.cv3d(torch.cat([x3d, x3], dim=1))))

            x2d = interpolate(x3d, data["ids32"])
            x2d = self.activation(self.bn2d(self.cv2d(torch.cat([x2d, x2], dim=1))))

            x1d = interpolate(x2d, data["ids21"])
            x1d = self.activation(self.bn1d(self.cv1d(torch.cat([x1d, x1], dim=1))))

            xout = interpolate(x1d, data["ids10"])
            xout = self.activation(self.bn0d(self.cv0d(torch.cat([xout, x0], dim=1))))
            xout = self.dropout(xout)
            if cat_in_last_layer is not None:
                xout = torch.cat([xout, cat_in_last_layer.expand(-1, -1, xout.shape[2])], dim=1)
            xout = self.fcout(xout)

        else:

            xout = x4
            xout = self.dropout(xout)
            xout = self.fcout(xout)
            xout = xout.mean(dim=2)

        return xout


if __name__ == '__main__':
    net = FKAConvNetwork(3, 3)
    input = {}
    input['x'] = torch.randn(2, 3, 3000)
    input['pos'] = torch.randn(2, 3, 3000)
    output = net(input, spectral_only=False)
    print(output.shape)