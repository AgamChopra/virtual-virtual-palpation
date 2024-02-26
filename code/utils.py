"""
Created on September 2023
@author: Agamdeep Chopra
@email: achopra4@uw.edu
@website: https://agamchopra.github.io/
@affiliation: University of Washington, Seattle, USA

@Refs:
    - PyTorch 2.0 stable documentation @ https://pytorch.org/docs/stable/
"""
import torch
import torch.nn as nn
from math import ceil
import matplotlib.pyplot as plt
from pytorch_msssim import SSIM


def pad3d(inpt, target):
    '''
    Pad or crop input image to match target size

    Parameters
    ----------
    inpt : torch.tensor
        Input tensor to be padded or cropped of shape (B, C, X, Y, Z).
    target : torch.tensor, tuple
        Target tensor of shape (B, C, X, Y, Z) or tuple of shape (X, Y, Z).

    Returns
    -------
    torch.tensor
        Resized (padded or cropped) input tensor matching size of target.

    '''
    if torch.is_tensor(target):
        delta = [target.shape[2+i] - inpt.shape[2+i] for i in range(3)]
    else:
        delta = [target[i] - inpt.shape[2+i] for i in range(3)]
    return nn.functional.pad(input=inpt, pad=(ceil(delta[2]/2),
                                              delta[2] - ceil(delta[2]/2),
                                              ceil(delta[1]/2),
                                              delta[1] - ceil(delta[1]/2),
                                              ceil(delta[0]/2),
                                              delta[0] - ceil(delta[0]/2)),
                             mode='constant', value=0.).to(dtype=inpt.dtype,
                                                           device=inpt.device)


def norm(x, mode='min-max', epsilon=1E-9):
    if mode == 'min-max':
        return (x - x.min()) / (x.max() - x.min() + epsilon)
    elif mode == 'max':
        return x / (x.max() + epsilon)
    elif mode == 'std':
        return (x - x.mean()) / (torch.std(x) + epsilon)


def show_images(in_data, num_samples=9, cols=3):
    data = torch.zeros(in_data.shape, requires_grad=False)

    data[:, :-1] = in_data[:, :-1].detach().cpu()
    data[:, -1] = norm(in_data[:, -1]).detach().cpu()

    data = data[..., int(data.shape[-1]/2)]

    plt.figure(figsize=(15, 15))
    for i, img in enumerate(data):
        if i == num_samples:
            break
        plt.subplot(int(num_samples/cols) + 1, cols, i + 1)
        plt.axis('off')
        if img.shape[0] == 1:
            plt.imshow(img[0], cmap='turbo')
        else:
            plt.imshow(img.permute(1, 2, 0))
    plt.show()


def getPositionEncoding(seq_len, d=64, n=10000):
    P = torch.zeros((seq_len, d))
    for k in range(seq_len):
        for i in torch.arange(int(d/2)):
            denominator = torch.pow(n, 2*i/d)
            P[k, 2*i] = torch.sin(k/denominator)
            P[k, 2*i+1] = torch.cos(k/denominator)
    return P


def get_cos_betas(steps, max_beta=0.999):
    def alpha_bar(t): return torch.cos((t + 0.008) / 1.008 * torch.pi / 2) ** 2
    i = torch.linspace(0, steps - 1, steps)
    t1 = i / steps
    t2 = (i + 1) / steps
    betas = 1 - alpha_bar(t2) / alpha_bar(t1)
    betas_clipped = torch.clamp(betas, 0., max_beta)
    return betas_clipped


def get_betas(steps=1000, scheduler='lin'):
    if scheduler == 'lin':
        scale = 1000 / steps
        start = scale * 0.0001
        end = scale * 0.02
        return torch.linspace(start, end, steps)
    elif scheduler == 'cos':
        return get_cos_betas(steps)
    else:
        raise NotImplementedError(f"scheduler not implemented: {scheduler}")


class Block(nn.Module):
    def __init__(self, ic=None, embd_dim=None, oc=None, hc=None, t_enc=True,
                 norm=nn.InstanceNorm3d):
        super(Block, self).__init__()
        self.t_enc = t_enc
        if hc is None:
            if self.t_enc:
                self.mlp = nn.Sequential(nn.Linear(embd_dim, oc), nn.ReLU())

            self.layer = nn.Sequential(nn.Conv3d(
                in_channels=ic, out_channels=oc, kernel_size=3),
                nn.ReLU(), norm(oc))

            self.out_block = nn.Sequential(nn.Conv3d(
                in_channels=oc, out_channels=oc, kernel_size=2),
                nn.ReLU(), norm(oc))
        else:
            if self.t_enc:
                self.mlp = nn.Sequential(nn.Linear(embd_dim, hc), nn.ReLU())

            self.layer = nn.Sequential(nn.Conv3d(
                in_channels=ic, out_channels=hc, kernel_size=3), nn.ReLU(),
                norm(hc))

            self.out_block = nn.Sequential(nn.Conv3d(in_channels=hc,
                                                     out_channels=hc,
                                                     kernel_size=2),
                                           nn.ReLU(), norm(hc),
                                           nn.ConvTranspose3d(in_channels=hc,
                                                              out_channels=oc,
                                                              kernel_size=2,
                                                              stride=2),
                                           nn.ReLU(), norm(oc))

    def forward(self, x, t=None):
        y = self.layer(x)

        if self.t_enc:
            t = self.mlp(t)
            t = t[(..., ) + (None, ) * 3]
            y = y + t

        y = self.out_block(y)
        return y


class Block2(nn.Module):
    def __init__(self, in_c, embd_dim, out_c, hid_c=None, final_layer=False):
        super(Block2, self).__init__()
        if hid_c is None:
            self.mlp = nn.Sequential(nn.utils.spectral_norm(
                nn.Linear(embd_dim, out_c)), nn.ReLU())

            self.layer = nn.Sequential(nn.utils.spectral_norm(nn.Conv3d(
                in_channels=in_c, out_channels=out_c, kernel_size=3)),
                nn.ReLU())

            self.out_block = nn.Sequential(nn.utils.spectral_norm(nn.Conv3d(
                in_channels=out_c, out_channels=out_c, kernel_size=2)),
                nn.ReLU())

            if final_layer:
                self.pool = False

            else:
                self.pool = True
                self.pool_block = nn.Sequential(nn.utils.spectral_norm(
                    nn.Conv3d(in_channels=out_c, out_channels=out_c,
                              kernel_size=2, stride=2)), nn.ReLU())

        else:
            self.pool = False

            self.mlp = nn.Sequential(nn.utils.spectral_norm(
                nn.Linear(embd_dim, hid_c)), nn.ReLU())

            self.layer = nn.Sequential(nn.utils.spectral_norm(nn.Conv3d(
                in_channels=in_c, out_channels=hid_c, kernel_size=3)),
                nn.ReLU())

            self.out_block = nn.Sequential(nn.utils.spectral_norm(
                nn.Conv3d(in_channels=hid_c, out_channels=hid_c,
                          kernel_size=2)),
                nn.ReLU(),
                nn.utils.spectral_norm(
                nn.ConvTranspose3d(
                    in_channels=hid_c,
                    out_channels=out_c,
                    kernel_size=2, stride=2)),
                nn.ReLU(),)

    def forward(self, x, embds):
        embds = self.mlp(embds)
        embds = embds[(..., ) + (None, ) * 3]
        y = self.layer(x)
        y = y + embds
        y = self.out_block(y)

        if self.pool:
            y_ = self.pool_block(y)
            return y_, y
        else:
            return y


class attention_grid(nn.Module):
    def __init__(self, x_c, g_c, i_c, stride=3, mode='trilinear'):
        super(attention_grid, self).__init__()
        self.input_filter = nn.utils.spectral_norm(nn.Conv3d(
            in_channels=x_c, out_channels=i_c, kernel_size=1, stride=stride,
            bias=False))

        self.gate_filter = nn.utils.spectral_norm(nn.Conv3d(
            in_channels=g_c, out_channels=i_c, kernel_size=1, stride=1,
            bias=True))

        self.tfilt = nn.utils.spectral_norm(nn.Conv3d(
            in_channels=i_c, out_channels=i_c, kernel_size=1, stride=1,
            bias=False))

        self.psi = nn.utils.spectral_norm(
            nn.Conv3d(in_channels=i_c, out_channels=1, kernel_size=1, stride=1,
                      bias=True))

        self.mode = mode

    def forward(self, x, g):
        a = nn.functional.relu(self.tfilt(self.input_filter(x)))
        b = self.gate_filter(g)

        if a.shape[-1] < b.shape[-1]:
            a = pad3d(a, b)

        elif a.shape[-1] > b.shape[-1]:
            b = pad3d(b, a)

        w = torch.sigmoid(self.psi(nn.functional.relu(a + b)))
        w = nn.functional.interpolate(w, size=x.shape[2:], mode=self.mode)

        y = x * w

        return y, w


class attention_grid2(nn.Module):
    def __init__(self, x_c, g_c, i_c, embd_dim, stride=3, mode='trilinear'):
        super(attention_grid2, self).__init__()
        self.input_filter = nn.utils.spectral_norm(nn.Conv3d(
            in_channels=x_c, out_channels=i_c, kernel_size=1, stride=stride,
            bias=False))

        self.gate_filter = nn.utils.spectral_norm(nn.Conv3d(
            in_channels=g_c, out_channels=i_c, kernel_size=1, stride=1,
            bias=True))

        self.mlp = nn.Sequential(nn.utils.spectral_norm(
            nn.Linear(embd_dim, i_c, bias=False)), nn.ReLU())

        self.tfilt = nn.utils.spectral_norm(nn.Conv3d(
            in_channels=i_c, out_channels=i_c, kernel_size=1, stride=1,
            bias=False))

        self.psi = nn.utils.spectral_norm(
            nn.Conv3d(in_channels=i_c, out_channels=1, kernel_size=1, stride=1,
                      bias=True))

        self.mode = mode

    def forward(self, x, g, t):
        t = self.mlp(t)
        t = t[(..., ) + (None, ) * 3]
        a = nn.functional.relu(self.tfilt(self.input_filter(x) + t))
        b = self.gate_filter(g)

        if a.shape[-1] < b.shape[-1]:
            a = pad3d(a, b)

        elif a.shape[-1] > b.shape[-1]:
            b = pad3d(b, a)

        w = torch.sigmoid(self.psi(nn.functional.relu(a + b)))
        w = nn.functional.interpolate(w, size=x.shape[2:], mode=self.mode)

        y = x * w

        return y, w


def grad_penalty(critic, real, fake, weight, input_signal):
    assert real.device == fake.device, "inputs must be on same device!"
    assert real.device == next(critic.parameters()).device, "inputs and critic\
        must be on same device!"
    b_size, c, h, w, d = real.shape
    epsilon = torch.rand((b_size, 1, 1, 1, 1),
                         device=real.device).repeat(1, c, h, w, d)
    interp_img = (real * epsilon) + (fake * (1 - epsilon))

    mix_score = critic(torch.cat([interp_img, input_signal], dim=1))

    grad = torch.autograd.grad(outputs=mix_score,
                               inputs=interp_img,
                               grad_outputs=torch.ones(mix_score.shape,
                                                       device=real.device),
                               create_graph=True,
                               retain_graph=True)[0]

    grad = grad.view(b_size, -1)
    grad_norm = torch.sqrt((torch.sum(grad ** 2, dim=1)) + 1E-12)
    penalty = torch.mean((grad_norm - 1) ** 2)
    return weight * penalty


class PSNR():
    def __init__(self, epsilon=1E-4):
        self.name = "PSNR"
        self.epsilon = epsilon

    def __call__(self, x, y):
        assert x.shape == y.shape, "inputs must be of same shape!"
        mse = torch.mean((x - y) ** 2)
        psnr = 20 * torch.log10(torch.max(x)) - 10 * torch.log10(mse)
        psnr = torch.clip(psnr, -48, 48)
        loss = (48 - psnr) * self.epsilon
        return loss


class ssim_loss(nn.Module):
    def __init__(self, channel=1, spatial_dims=3, win_size=11, win_sigma=1.5):
        super(ssim_loss, self).__init__()
        self.ssim = SSIM(channel=channel, spatial_dims=spatial_dims,
                         win_size=win_size, win_sigma=win_sigma)

    def forward(self, x, y):
        assert x.shape == y.shape, "inputs must be of same shape!"
        loss = 1 - self.ssim(x, y)
        return loss


def get_random_mask(x, mod, n_encodings, N, TAU=1E-3):  # N-1 possible pairs
    assert x.device == mod.device, "inputs must be on same device!"
    assert x.device == n_encodings.device, "inputs must be on same device!"
    assert x.shape == mod.shape, "inputs must be of same shape!"
    assert x.dtype == mod.dtype, "inputs must be of same dtype!"

    idx = torch.randint(0, N-1, (x.shape[0],), device=x.device)
    idx_0 = idx[(..., ) + (None, ) * 4] / N
    idx_1 = (idx[(..., ) + (None, ) * 4] + 1) / N

    a = x > idx_0
    a = (a * torch.where(idx_0 > 0., 1, 0)) + (torch.where(mod > TAU, 1, 0) *
                                               torch.where(idx_0 > 0., 0, 1))
    b = x > idx_1

    x_t0 = a * x
    x_t1 = b * x
    t1 = n_encodings[idx]

    assert x_t0.dtype == x.dtype
    assert x_t1.dtype == x.dtype
    assert x_t0.shape == x.shape
    assert x_t1.shape == x.shape

    return x_t0, x_t1, t1


class mask_generator():
    def __init__(self, N=20, encoding_dim=128, TAU=1E-3):
        self.N = N
        self.n_encode = getPositionEncoding(N-1, encoding_dim)
        self.TAU = TAU

    def apply(self, x0, T2):
        assert x0.device == T2.device, "inputs must be on same device!"
        assert x0.shape == T2.shape, "inputs must be of same shape!"
        if self.n_encode.device != x0.device:
            self.n_encode = self.n_encode.to(device=x0.device)
        return get_random_mask(x0, T2, self.n_encode, self.N, self.TAU)


def compose(phis):
    return torch.mean(phis.to(dtype=torch.float), dim=1)


def pet_norm(raw_pet):
    '''
    a = torch.flatten(raw_pet, start_dim=1)
    mins = torch.min(a, dim=-1).values
    norm_pet = raw_pet + mins
    a = torch.flatten(norm_pet, start_dim=1)
    maxs = torch.max(a, dim=-1).values
    norm_pet /= maxs
    return norm_pet
    '''
    return raw_pet


def pix_error(x, y):
    '''
    Pixel/Voxel level mean squared error of non empty regions of input x

    Parameters
    ----------
    x : torch tensor
        input referance/target/known tensor.
    y : torch tensor
        input tensor being compared/evaluated.

    Returns
    -------
    err : float
        average error.

    '''
    a = torch.flatten(x, start_dim=1)
    mins = torch.min(a, dim=-1).values
    mask = torch.where(x > mins, 0, 1)
    n = torch.sum(mask)
    err = torch.sum(mask * (x - y) ** 2) / n
    return err


def per_error(x, y):
    '''
    Pixel/Voxel level % error of non empty regions of input x

    Parameters
    ----------
    x : torch tensor
        input referance/target/known tensor.
    y : torch tensor
        input tensor being compared/evaluated.

    Returns
    -------
    err : float
        average error.

    '''
    mask = torch.where(x > 0, 1, 0)
    n = torch.sum(mask)
    y_masked = y * mask
    abs_err = torch.abs(x - y_masked)
    err = mask * 100 * abs_err / (x + 1E-9)
    err = torch.sum(err) / (n + 1E-9)
    return err


def distributions(x, y, bins=50, rng=(0, 1)):
    plt.figure(figsize=(10, 5), dpi=500)

    idx = torch.linspace(rng[0], rng[1], bins)

    d1 = torch.histc(x, bins=bins, min=rng[0], max=rng[1])
    d2 = torch.histc(y, bins=bins, min=rng[0], max=rng[1])

    d1 /= d1.max()
    d2 /= d2.max()

    plt.plot(idx, d1, 'k-', label='Target')
    plt.plot(idx, d2, 'r-', label='Predicted')
    plt.legend()
    plt.title(f'n = {x.shape[0]}')
    plt.show()
