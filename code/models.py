"""
Created on October 2023
@author: Agamdeep Chopra
@email: achopra4@uw.edu
@website: https://agamchopra.github.io/
@affiliation: KurtLab, Department of Mechanical Engineering,
              University of Washington, Seattle, USA
@Refs:
    - PyTorch 2.0 stable documentation @ https://pytorch.org/docs/stable/
"""
import torch
import torch.nn as nn
from utils import pad3d, attention_grid, Block, Block2


# UNet
class Unet(nn.Module):
    def __init__(self, CH_IN=5, CH_OUT=1, n=1, p=0.3):
        super(Unet, self).__init__()
        print('Model size factor =', int(64/n))
        self.drop = nn.Dropout3d(p)
        self.layer1 = nn.Sequential(
            nn.Conv3d(CH_IN, int(64/n), 2, 1), nn.ReLU(),
            nn.InstanceNorm3d(int(64/n)))

        self.layer2 = Block(ic=int(64/n), oc=int(128/n), t_enc=False)

        self.layer3 = Block(ic=int(128/n), oc=int(256/n), t_enc=False)

        self.layer4 = Block(ic=int(256/n), oc=int(512/n), t_enc=False)

        self.layer5 = Block(ic=int(512/n), oc=int(512/n),
                            hc=int(1024/n), t_enc=False)

        self.pool2 = nn.Sequential(nn.Conv3d(in_channels=int(
            128/n), out_channels=int(128/n), kernel_size=2, stride=2),
            nn.ReLU(), nn.InstanceNorm3d(int(128/n)))

        self.pool3 = nn.Sequential(nn.Conv3d(in_channels=int(
            256/n), out_channels=int(256/n), kernel_size=2, stride=2),
            nn.ReLU(), nn.InstanceNorm3d(int(256/n)))

        self.pool4 = nn.Sequential(nn.Conv3d(in_channels=int(
            512/n), out_channels=int(512/n), kernel_size=2, stride=2),
            nn.ReLU(), nn.InstanceNorm3d(int(512/n)))

        self.layer6 = Block(ic=int(1024/n), t_enc=False,
                            oc=int(256/n), hc=int(512/n))

        self.layer7 = Block(ic=int(512/n), t_enc=False,
                            oc=int(128/n), hc=int(256/n))

        self.layer8 = Block(ic=int(256/n), t_enc=False, oc=int(64/n))

        self.out = nn.Sequential(nn.Conv3d(in_channels=int(64/n),
                                           out_channels=int(64/n),
                                           kernel_size=1),
                                 nn.ReLU(), nn.InstanceNorm3d(int(64/n)),
                                 nn.Conv3d(in_channels=int(64/n),
                                           out_channels=CH_OUT, kernel_size=1))

    def forward(self, x, pad2size=188):
        assert len(x.shape) == 5, "Expected input to be a 5D tensor of shap\
            e (N,C,x,y,z)"
        assert x[0].device == self.pool2[0].weight.device, "'x', 'paramet\
            ers' are expected to be on the same device, but found at least two\
                devices, '%s' and '%s'" % (x[0].device, self.out.weight.device)
        x_ = pad3d(x, (pad2size, pad2size, pad2size))
        y = self.drop(self.layer1(x_))

        y2 = self.drop(self.layer2(y))
        y = self.drop(self.pool2(y2))

        y3 = self.drop(self.layer3(y))
        y = self.drop(self.pool3(y3))

        y4 = self.drop(self.layer4(y))
        y = self.drop(self.pool4(y4))

        y = self.drop(self.layer5(y))

        y = torch.cat((y4, pad3d(y, y4)), dim=1)
        y = self.drop(self.layer6(y))

        y = torch.cat((y3, pad3d(y, y3)), dim=1)
        y = self.drop(self.layer7(y))

        y = torch.cat((y2, pad3d(y, y2)), dim=1)
        y = self.drop(self.layer8(y))

        y = pad3d(y, x)

        y = self.out(y)

        return y


def test_unet(device='cpu', N=1):
    a = torch.ones((N, 5, 150, 150, 150), device=device)

    model = Unet(5, 1, 32).to(device)

    b = model(a)

    print(a.shape)
    print(b.shape)


# Attention-UNet
class AUnet(nn.Module):
    def __init__(self, CH_IN=5, CH_OUT=1, n=1, p=0.3,
                 attention_embed=128, mode='trilinear'):
        super(AUnet, self).__init__()
        print('Model size factor =', int(64/n))
        self.drop = nn.Dropout3d(p)
        self.layer1 = nn.Sequential(
            nn.Conv3d(CH_IN, int(64/n), 2, 1), nn.ReLU(),
            nn.InstanceNorm3d(int(64/n)))

        self.layer2 = Block(ic=int(64/n), oc=int(128/n), t_enc=False)

        self.layer3 = Block(ic=int(128/n), oc=int(256/n), t_enc=False)

        self.layer4 = Block(ic=int(256/n), oc=int(512/n), t_enc=False)

        self.layer5 = Block(ic=int(512/n), oc=int(512/n),
                            hc=int(1024/n), t_enc=False)

        self.pool2 = nn.Sequential(nn.Conv3d(in_channels=int(
            128/n), out_channels=int(128/n), kernel_size=2, stride=2),
            nn.ReLU(), nn.InstanceNorm3d(int(128/n)))

        self.pool3 = nn.Sequential(nn.Conv3d(in_channels=int(
            256/n), out_channels=int(256/n), kernel_size=2, stride=2),
            nn.ReLU(), nn.InstanceNorm3d(int(256/n)))

        self.pool4 = nn.Sequential(nn.Conv3d(in_channels=int(
            512/n), out_channels=int(512/n), kernel_size=2, stride=2),
            nn.ReLU(), nn.InstanceNorm3d(int(512/n)))

        self.layer6 = Block(ic=int(1024/n), t_enc=False,
                            oc=int(256/n), hc=int(512/n))

        self.layer7 = Block(ic=int(512/n), t_enc=False,
                            oc=int(128/n), hc=int(256/n))

        self.layer8 = Block(ic=int(256/n), t_enc=False, oc=int(64/n))

        self.out = nn.Sequential(nn.Conv3d(in_channels=int(64/n),
                                           out_channels=int(64/n),
                                           kernel_size=1),
                                 nn.ReLU(), nn.InstanceNorm3d(int(64/n)),
                                 nn.Conv3d(in_channels=int(64/n),
                                           out_channels=CH_OUT, kernel_size=1))

        self.attention3 = attention_grid(
            int(128/n), int(128/n), int(128/n),
            embd_dim=attention_embed, mode=mode)

        self.attention2 = attention_grid(
            int(256/n), int(256/n), int(256/n),
            embd_dim=attention_embed, mode=mode)

        self.attention1 = attention_grid(
            int(512/n), int(512/n), int(512/n),
            embd_dim=attention_embed, mode=mode)

    def forward(self, x, pad2size=188, get_maps=False):
        assert len(x.shape) == 5, "Expected input to be a 5D tensor of shap\
            e (N,C,x,y,z)"
        assert x[0].device == self.pool2[0].weight.device, "'x', 'paramet\
            ers' are expected to be on the same device, but found at least two\
                devices, '%s' and '%s'" % (x[0].device, self.out.weight.device)
        x_ = pad3d(x, (pad2size, pad2size, pad2size))
        y = self.drop(self.layer1(x_))

        y2 = self.drop(self.layer2(y))
        y = self.drop(self.pool2(y2))

        y3 = self.drop(self.layer3(y))
        y = self.drop(self.pool3(y3))

        y4 = self.drop(self.layer4(y))
        y = self.drop(self.pool4(y4))

        y = self.drop(self.layer5(y))

        y4, m1 = self.attention1(y4, y)
        y = torch.cat((y4, pad3d(y, y4)), dim=1)
        y = self.drop(self.layer6(y))

        y3, m2 = self.attention2(y3, y)
        y = torch.cat((y3, pad3d(y, y3)), dim=1)
        y = self.drop(self.layer7(y))

        y2, m3 = self.attention3(y2, y)
        y = torch.cat((y2, pad3d(y, y2)), dim=1)
        y = self.drop(self.layer8(y))

        y = pad3d(y, x)

        y = self.out(y)

        if get_maps:
            return y, (m1, m2, m3)
        else:
            return y


def test_aunet(device='cpu', N=1):
    a = torch.ones((N, 5, 150, 150, 150), device=device)

    model = AUnet(5, 1, 32, 0.3, 8).to(device)

    b = model(a)

    print(model.device())
    print(a.shape)
    print(b.shape)


# Tau_Net: Threshold-Attention-UNet
class TAUnet(nn.Module):
    def __init__(self, in_c, out_c, embd_dim, n=1, mode='trilinear'):
        super(TAUnet, self).__init__()
        n = int(64 / n)
        print('Model size factor = %d' % n)
        self.mode = mode
        self.out_c = out_c

        self.layer1 = Block2(in_c=in_c,
                             embd_dim=embd_dim, out_c=int(1 * n))

        self.layer2 = Block2(in_c=int(1 * n),
                             embd_dim=embd_dim, out_c=int(2 * n))

        self.layer3 = Block2(in_c=int(2 * n),
                             embd_dim=embd_dim, out_c=int(4 * n))

        self.layer4 = Block2(in_c=int(4 * n),
                             embd_dim=embd_dim, out_c=int(8 * n))

        self.layer5 = Block2(in_c=int(8 * n), embd_dim=embd_dim,
                             out_c=int(8 * n), hid_c=int(16 * n))

        self.layer6 = Block2(in_c=int(16 * n), embd_dim=embd_dim,
                             out_c=int(4 * n), hid_c=int(8 * n))

        self.layer7 = Block2(in_c=int(8 * n), embd_dim=embd_dim,
                             out_c=int(2 * n), hid_c=int(4 * n))

        self.layer8 = Block2(in_c=int(4 * n), embd_dim=embd_dim,
                             out_c=int(1 * n), hid_c=int(2 * n))

        self.layer9 = Block2(in_c=int(2 * n), embd_dim=embd_dim,
                             out_c=int(1 * n), final_layer=True)

        self.out = nn.Conv3d(in_channels=int(
            1 * n), out_channels=out_c, kernel_size=1)

        self.skip1 = attention_grid(
            int(1 * n), int(1 * n), int(1 * n), embd_dim)

        self.skip2 = attention_grid(
            int(2 * n), int(2 * n), int(2 * n), embd_dim)

        self.skip3 = attention_grid(
            int(4 * n), int(4 * n), int(4 * n), embd_dim)

        self.skip4 = attention_grid(
            int(8 * n), int(8 * n), int(8 * n), embd_dim)

        self.attention_mask = None

    def forward(self, x, embds):
        assert x.device == embds.device, "inputs 'x' and 'embds' are expected \
            to be on the same device, but found at least two devices, '%s' and\
                '%s'!" % (
            x.device, embds.device)

        assert x.device == self.out.weight.device, "inputs 'x' and 'parameters\
            ' are expected to be on the same device, but found at least two de\
                vices, '%s' and '%s'!" % (
            x.device, self.out.weight.device)

        assert len(
            x.shape) == 5, "Expected input to be a 5D tensor of shape (N,C,x,y\
            ,z)"

        assert len(
            embds.shape) == 2, "Expected embds to be a 2D tensor (N,m)"

        assert x.shape[0] == embds.shape[0], "Batch size of input must match b\
            atch size of embds!"

        assert x.dtype == torch.float, "input must be of type torch.float!"

        assert embds.dtype == torch.float, "embds must be of type torch.float!"

        x_ = pad3d(x, (186, 186, 186)).to(dtype=torch.float)

        y, y1 = self.layer1(x_, embds)

        y, y2 = self.layer2(y, embds)

        y, y3 = self.layer3(y, embds)

        y, y4 = self.layer4(y, embds)

        y = self.layer5(y, embds)
        y4, _ = self.skip4(y4, y, embds)

        y = torch.cat((y4, pad3d(y, y4)), dim=1)
        y = self.layer6(y, embds)
        y3, _ = self.skip3(y3, y, embds)

        y = torch.cat((y3, pad3d(y, y3)), dim=1)
        y = self.layer7(y, embds)
        y2, _ = self.skip2(y2, y, embds)

        y = torch.cat((y2, pad3d(y, y2)), dim=1)
        y = self.layer8(y, embds)
        y1, self.attention_mask = self.skip1(y1, y, embds)

        y = torch.cat((y1, pad3d(y, y1)), dim=1)
        y = self.layer9(y, embds)

        y = nn.functional.sigmoid(self.out(y))

        return y


def get_models(CH_IN, CH_OUT, n):
    models = {'unet': lambda: Unet(CH_IN, CH_OUT, n),
              'aunet': lambda: AUnet(CH_IN, CH_OUT, n),
              'taunet': lambda: TAUnet(CH_IN, CH_OUT, 64, n)}
    return models


if __name__ == '__main__':
    devices = ['cpu', 'cuda']
    try:
        [test_unet(device, 1) for device in devices]
        print('###unet passed')
    except Exception:
        print('***unet failed')
    try:
        [test_aunet(device, 1) for device in devices]
        print('###aunet passed')
    except Exception:
        print('***aunet failed')
