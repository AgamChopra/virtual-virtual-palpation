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
from utils import Block, pad3d


# UNet for training task 1, 2, and 3
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
    a = torch.ones((N, 5, 180, 180, 180), device=device)

    model = Unet(5, 1, 32).to(device)

    b = model(a)

    print(a.shape)
    print(b.shape)


if __name__ == '__main__':
    try:
        test_unet('cuda', 1)
        print('###unet passed')
    except Exception:
        test_unet('cuda', 1)
        print('***unet failed')
