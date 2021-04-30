import torch
from torch import nn


class ResBlk(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(ResBlk, self).__init__()
        self.resBlk = nn.Sequential(
            nn.Conv2d(in_chan, out_chan, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(out_chan, affine=True, track_running_stats=True),
            nn.ReLU(True),
            nn.Conv2d(out_chan, out_chan, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(out_chan, affine=True, track_running_stats=True)
        )

    def forward(self, x):
        out = x + self.resBlk(x)

        return out

class Encoder(nn.Module):

    def __init__(self, first_dim=64, res_block=6):
        super(Encoder, self).__init__()

        layers = []
        layers.append(nn.Conv2d(3, first_dim, 7, 1, 3, bias=False))
        layers.append(nn.InstanceNorm2d(first_dim, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(True))

        curr_dim = first_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim * 2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(True))
            curr_dim = curr_dim*2

        for i in range(res_block):
            layers.append(ResBlk(curr_dim, curr_dim))

        self.convs = nn.Sequential(*layers)

        #self.convs.type(dst_type=torch.float16)
        #self.layers.type(dst_type=torch.float16)

    def forward(self, x):
        out = self.convs(x)

        return out


class Generator(nn.Module):
    def __init__(self, zc, dim=2048):
        super(Generator, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(zc, dim, 4, 1, 0, bias=False),
            nn.InstanceNorm2d(dim, affine=True, track_running_stats=True),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, dim // 2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(dim // 2, affine=True, track_running_stats=True),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim // 2, dim // 4, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(dim // 4, affine=True, track_running_stats=True),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim // 4, dim // 8, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(dim // 8, affine=True, track_running_stats=True),
            nn.ReLU(True)
        )
        #self.deconv.type(dst_type=torch.float16)

    def forward(self, x):
        out = self.deconv(x)
        return out


class Transformer(nn.Module):
    def __init__(self, attr_dim, dim=256, res_block=6):
        super(Transformer, self).__init__()

        layers = []
        layers.append(nn.Conv2d(dim + attr_dim, dim, 3, 1, 1, bias=False))
        layers.append(nn.InstanceNorm2d(dim, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(True))

        for i in range(res_block):
            layers.append(ResBlk(dim, dim))

        self.trans = nn.Sequential(*layers)

        self.mask = nn.Sequential(
            nn.Conv2d(dim, 1, 7, 1, 3, bias=False),
            nn.Tanh()
        )

        #self.trans.type(dst_type=torch.float16)
        #self.resBlk.type(dst_type=torch.float16)
        #self.mask.type(dst_type=torch.float16)

    def forward(self, x, c):
        x_shape = x.size()
        c_shape = c.size()

        attr_vec = c.view((c_shape[0], c_shape[1], 1))
        c = attr_vec.expand((c_shape[0], c_shape[1], x_shape[2] * x_shape[3]))
        c = c.reshape((c_shape[0], c_shape[1], x_shape[2], x_shape[3]))

        out = torch.cat([x, c], dim=1)
        out = self.trans(out)

        mask = self.mask(out)
        _mask = (1 + mask) / 2

        result = _mask * out + mask * x

        return result


class Reconstructor(nn.Module):
    def __init__(self, dim=256):
        super(Reconstructor, self).__init__()

        self.deconvs = nn.Sequential(
            nn.ConvTranspose2d(dim, dim//2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(dim//2, affine=True, track_running_stats=True),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim//2, dim//4, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(dim//4, affine=True, track_running_stats=True),
            nn.ReLU(True)
        )
        self.convs = nn.Sequential(
            nn.Conv2d(dim//4, 3, 7, 1, 3, bias=False),
            nn.Tanh()
        )

        #self.deconv.type(dst_type=torch.float16)

    def forward(self, x):
        out = self.deconvs(x)
        out = self.convs(out)

        return out


class Discriminator(nn.Module):
    def __init__(self, attr_dim, dim=64):
        super(Discriminator, self).__init__()

        self.convs = nn.Sequential(
            nn.Conv2d(3, dim, 4, 2, 1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(dim, dim * 2, 4, 2, 1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(dim * 2, dim * 4, 4, 2, 1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(dim * 4, dim * 8, 4, 2, 1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(dim * 8, dim * 16, 4, 2, 1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(dim * 16, dim * 32, 4, 2, 1),
            nn.LeakyReLU(0.01)
        )
        self.output = nn.Sequential(
            nn.Conv2d(dim * 32, 1, 3, 1, 1, bias=False),
        )
        self.labels = nn.Sequential(
            nn.Conv2d(dim * 32, attr_dim, 2, 1, 0, bias=False),
        )

        #self.convs.type(dst_type=torch.float16)
        #self.output.type(dst_type=torch.float16)
        #self.labels.type(dst_type=torch.float16)

    def forward(self, x):
        out = self.convs(x)
        result = self.output(out)
        label = self.labels(out)

        return result, label.view(label.size(0), label.size(1))
