import torch
import torch.nn as nn
import torch.nn.functional as F

class PointNet(nn.Module):

    def __init__(self, in_channels,
            out_channels,
            hidden_dim=32, segmentation=False, **kwargs):
        super().__init__()
        
        self.fc_in = nn.Conv1d(in_channels+3, 2*hidden_dim, 1)

        mlp_layers = [nn.Conv1d(2*hidden_dim, hidden_dim, 1) for _ in range(3)]
        self.mlp_layers = nn.ModuleList(mlp_layers)

        self.fc_3 = nn.Conv1d(2*hidden_dim, hidden_dim, 1)

        self.segmentation=segmentation

        if segmentation:
            self.fc_out = nn.Conv1d(2*hidden_dim, out_channels, 1)
        else:
            self.fc_out = nn.Linear(hidden_dim, out_channels)

        self.activation = nn.ReLU()

    def forward_spatial(self, data):
        return {}


    def forward(self, data, spatial_only=False, spectral_only=False, cat_in_last_layer=None):
        
        if spatial_only:
            return self.forward_spatial(data)

        if not spectral_only:
            spatial_data = self.forward_spatial(data)
            for key, value in spatial_data.items():
                data[key] = value
            # data = {**data, **spatial_data}


        x = data["x"]
        pos = data["pos"]

        x = torch.cat([x, pos], dim=1)

        x = self.fc_in(x)

        for i, l in enumerate(self.mlp_layers):
            x = l(self.activation(x))
            x_pool = torch.max(x, dim=2, keepdim=True)[0].expand_as(x)
            x = torch.cat([x, x_pool], dim=1)

        x = self.fc_3(self.activation(x))

        if self.segmentation:
            x_pool = torch.max(x, dim=2, keepdim=True)[0].expand_as(x)
            x = torch.cat([x, x_pool], dim=1)
        else:
            x = torch.max(x, dim=2)[0]
        
        x = self.fc_out(x)

        return x


class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, hidden_dim):
        super().__init__()

        # Submodules
        self.fc_0 = nn.Conv1d(in_channels, hidden_dim, 1)
        self.fc_1 = nn.Conv1d(hidden_dim, out_channels, 1)
        self.activation = nn.ReLU()

        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels,1)
        else:
            self.shortcut = nn.Identity()

        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        x_short = self.shortcut(x)
        x = self.fc_0(x)
        x = self.fc_1(self.activation(x))
        x = self.activation(x + x_short)
        return x

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
        x_q = self.q_conv(x).permute(0, 2, 1) # b, n, c
        x_k = self.k_conv(x)# b, c, n
        x_v = self.v_conv(x)
        energy = x_q @ x_k # b, n, n
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))
        x_r = x_v @ attention # b, c, n
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
        self.conv_reduction = nn.Conv1d(channels_high, channels_low, kernel_size=1, padding=0, bias=False)
        self.bn_reduction = nn.BatchNorm1d(channels_low)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, fms_high, fms_low):
        """
        Use the high level features with abundant catagory information to weight the low level features with pixel
        localization information. In the meantime, we further use mask feature maps with catagory-specific information
        to localize the mask position.
        :param fms_high: Features of high level. Tensor.
        :param fms_low: Features of low level.  Tensor.
        """
        b, c, h = fms_high.shape

        fms_high_gp = nn.AvgPool1d(fms_high.shape[2:])(fms_high).view(len(fms_high), c, 1)
        fms_high_gp = self.conv1x1(fms_high_gp)
        fms_high_gp = self.bn_high(fms_high_gp)
        fms_high_gp = self.relu(fms_high_gp)
        # fms_low_mask = torch.cat([fms_low, fm_mask], dim=1)
        fms_low_mask = self.conv3x3(fms_low)
        fms_low_mask = self.bn_low(fms_low_mask)
        fms_att = fms_low_mask * fms_high_gp
        out = self.relu(self.bn_reduction(self.conv_reduction(fms_high)) + fms_att)
        return out

class ResidualPointNet(nn.Module):
    ''' PointNet-based encoder network with ResNet blocks.
    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
    '''

    def __init__(self, in_channels, out_channels, hidden_dim=128, segmentation=False, **kwargs):
        super().__init__()

        self.fc_in = nn.Conv1d(in_channels+3, 2*hidden_dim, 1)
        self.block_0 = ResidualBlock(2*hidden_dim, hidden_dim, hidden_dim)
        self.sa_0 = SA_Layer(2*hidden_dim)
        self.block_1 = ResidualBlock(2*hidden_dim, hidden_dim, hidden_dim)
        self.sa_1 = SA_Layer(2 * hidden_dim)
        self.gca_1 = GCA(2*hidden_dim,2*hidden_dim)

        self.block_2 = ResidualBlock(2*hidden_dim, hidden_dim, hidden_dim)
        self.sa_2 = SA_Layer(2 * hidden_dim)
        self.gca_2 = GCA(2 * hidden_dim, 2 * hidden_dim)

        self.block_3 = ResidualBlock(2*hidden_dim, hidden_dim, hidden_dim)
        self.sa_3 = SA_Layer(2 * hidden_dim)
        self.gca_3 = GCA(2 * hidden_dim, 2 * hidden_dim)

        self.block_4 = ResidualBlock(2*hidden_dim, hidden_dim, hidden_dim)
        self.sa_4 = SA_Layer(2 * hidden_dim)

        self.segmentation = segmentation
        if self.segmentation:
            self.fc_out = nn.Conv1d(2*hidden_dim, out_channels, 1)
        else:
            self.fc_out = nn.Linear(hidden_dim, out_channels)


    def forward_spatial(self, data):
        return {}

    def forward(self, data, spatial_only=False, spectral_only=False, cat_in_last_layer=None):
        
        if spatial_only:
            return self.forward_spatial(data)

        if not spectral_only:
            spatial_data = self.forward_spatial(data)
            for key, value in spatial_data.items():
                data[key] = value
            # data = {**data, **spatial_data}
        x = data["x"]
        pos = data["pos"]
        x = torch.cat([x, pos], dim=1)

        x = self.fc_in(x)
        print(x.shape)
        
        x = self.block_0(x)
        x_pool = torch.max(x, dim=2, keepdim=True)[0].expand_as(x)
        x = torch.cat([x, x_pool], dim=1)
        print(x.shape)
        x0 = self.sa_0(x)
        
        x = self.block_1(x0)
        x_pool = torch.max(x, dim=2, keepdim=True)[0].expand_as(x)
        x = torch.cat([x, x_pool], dim=1)
        x1 = self.sa_1(x)
        print(x.shape)

        x01 = self.gca_1(x1,x0)
        
        x = self.block_2(x1)
        x_pool = torch.max(x, dim=2, keepdim=True)[0].expand_as(x)
        x = torch.cat([x, x_pool], dim=1)
        print(x.shape)
        x_fusion = x+x01
        x2 = self.sa_2(x_fusion)

        x12 = self.gca_2(x2,x1)
        
        x = self.block_3(x2)
        print(x.shape)
        x_pool = torch.max(x, dim=2, keepdim=True)[0].expand_as(x)
        x = torch.cat([x, x_pool], dim=1)
        x_fusion = x + x12
        x3 = self.sa_3(x_fusion)

        x23 = self.gca_3(x3,x2)

        x = self.block_4(x)
        print(x.shape)
        if self.segmentation:
            x_pool = torch.max(x, dim=2, keepdim=True)[0].expand_as(x)
            x = torch.cat([x, x_pool], dim=1)
            x_fusion = x+x23
            x4 = self.sa_4(x_fusion)
        else:
            x4 = torch.max(x, dim=2)[0]
        print(x4.shape)
        x = self.fc_out(x4)
        print(x.shape)
        return x

if __name__ == '__main__':
    net=ResidualPointNet(3,3)
    input={}
    input['x']=torch.randn(10,3,32)
    input['pos']=torch.randn(10,3,32)
    output=net(input)
    print(output.shape)