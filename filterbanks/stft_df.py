import math
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import os
import signal
import librosa

def init_kernel(frame_len,
                frame_hop,
                num_fft=None,
                window="sqrt_hann"):
    if window != "sqrt_hann":
        raise RuntimeError("Now only support sqrt hanning window in order "
                           "to make signal perfectly reconstructed")
    if not num_fft:
        # FFT points
        fft_size = 2 ** math.ceil(math.log2(frame_len))
    else:
        fft_size = num_fft
    # window [window_length]
    window = torch.hann_window(frame_len) ** 0.5
    S_ = 0.5 * (fft_size * fft_size / frame_hop) ** 0.5
    # window_length, F, 2 (real+imag)
    kernel = torch.rfft(torch.eye(fft_size) / S_, 1)[:frame_len]
    # 2, F, window_length
    kernel = torch.transpose(kernel, 0, 2) * window
    # 2F, 1, window_length
    kernel = torch.reshape(kernel, (fft_size + 2, 1, frame_len))
    return kernel


class STFTBase(nn.Module):
    """
    Base layer for (i)STFT
    NOTE:
        1) Recommend sqrt_hann window with 2**N frame length, because it
           could achieve perfect reconstruction after overlap-add
        2) Now haven't consider padding problems yet
    """

    def __init__(self,
                 frame_len,
                 frame_hop,
                 window="sqrt_hann",
                 num_fft=None):
        super(STFTBase, self).__init__()
        K = init_kernel(
            frame_len,
            frame_hop,
            num_fft=num_fft,
            window=window)
        self.K = nn.Parameter(K, requires_grad=False)
        self.stride = frame_hop
        self.window = window

    def freeze(self):
        self.K.requires_grad = False

    def unfreeze(self):
        self.K.requires_grad = True

    def check_nan(self):
        num_nan = torch.sum(torch.isnan(self.K))
        if num_nan:
            raise RuntimeError(
                "detect nan in STFT kernels: {:d}".format(num_nan))

    def extra_repr(self):
        return "window={0}, stride={1}, requires_grad={2}, kernel_size={3[0]}x{3[2]}".format(
            self.window, self.stride, self.K.requires_grad, self.K.shape)


class STFT(STFTBase):
    """
    Short-time Fourier Transform as a Layer
    """

    def __init__(self, *args, **kwargs):
        super(STFT, self).__init__(*args, **kwargs)

    def forward(self, x):
        """
        Accept raw waveform and output magnitude and phase
        x: input signal, N x 1 x S or N x S
        m: magnitude, N x F x T
        p: phase, N x F x T
        """
        if x.dim() not in [2, 3]:
            raise RuntimeError("Expect 2D/3D tensor, but got {:d}D".format(
                x.dim()))
        self.check_nan()
        # if N x S, reshape N x 1 x S
        if x.dim() == 2:
            x = torch.unsqueeze(x, 1)
        # N x 2F x T
        c = F.conv1d(x, self.K, stride=self.stride, padding=0)
        # N x F x T
        r, i = torch.chunk(c, 2, dim=1)
        m = (r ** 2 + i ** 2) ** 0.5
        p = torch.atan2(i, r)
        return m, p



class iSTFT(STFTBase):
    """
    Inverse Short-time Fourier Transform as a Layer
    """

    def __init__(self, *args, **kwargs):
        super(iSTFT, self).__init__(*args, **kwargs)

    def forward(self, m, p, squeeze=False):
        """
        Accept phase & magnitude and output raw waveform
        m, p: N x F x T
        s: N x C x S
        """
        if p.dim() != m.dim() or p.dim() not in [2, 3]:
            raise RuntimeError("Expect 2D/3D tensor, but got {:d}D".format(
                p.dim()))
        self.check_nan()
        # if F x T, reshape 1 x F x T
        if p.dim() == 2:
            p = torch.unsqueeze(p, 0)
            m = torch.unsqueeze(m, 0)
        r = m * torch.cos(p)
        i = m * torch.sin(p)
        # N x 2F x T
        c = torch.cat([r, i], dim=1)
        # N x 2F x T
        s = F.conv_transpose1d(c, self.K, stride=self.stride, padding=0)
        if squeeze:
            s = torch.squeeze(s)
        return s


class iSTFT_ri(STFTBase):
    """
    Inverse Short-time Fourier Transform as a Layer
    """

    def __init__(self, *args, **kwargs):
        super(iSTFT_ri, self).__init__(*args, **kwargs)

    def forward(self, r, i, squeeze=False):
        # N x 2F x T
        c = torch.cat([r, i], dim=1)
        # N x 2F x T
        s = F.conv_transpose1d(c, self.K, stride=self.stride, padding=0)
        if squeeze:
            s = torch.squeeze(s)
        return s

class ChannelWiseLayerNorm(nn.LayerNorm):
    """
    Channel wise layer normalization
    """

    def __init__(self, *args, **kwargs):
        super(ChannelWiseLayerNorm, self).__init__(*args, **kwargs)

    def forward(self, x):
        """
        x: BS x N x K
        """
        if x.dim() != 3:
            raise RuntimeError("{} accept 3D tensor as input".format(
                self.__name__))
        # BS x N x K => BS x K x N
        x = torch.transpose(x, 1, 2)
        x = super(ChannelWiseLayerNorm, self).forward(x)
        x = torch.transpose(x, 1, 2)
        return x


class DFComputer(nn.Module):
    def __init__(self,
                 frame_len=320,
                 frame_hop=160,
                 in_feature=['LPS'],
                 merge_mode='sum',
                 speaker_feature_dim=1,
                 cosIPD=True,
                 sinIPD=True,
                 n_filter = -1):
        super(DFComputer, self).__init__()
        self.cosIPD = cosIPD
        self.sinIPD = sinIPD
        # self.init_mic_pos()
   
        self.input_feature = in_feature
        self.spk_fea_dim = speaker_feature_dim
        # self.spk_fea_merge_mode = merge_mode
 
        if n_filter == -1:
            self.num_bins = frame_len//2 + 1
            self.stft = STFT(frame_len=frame_len, frame_hop=frame_hop, num_fft=frame_len)
        else:
            self.num_bins = n_filter
            self.stft = STFT(frame_len=frame_len, frame_hop=frame_hop, num_fft= (n_filter-1)*2)

        # print(self.num_bins)
        self.epsilon = 1e-8

        # calculate DF dimension
        self.df_dim = 0

       

        if 'LPS' in self.input_feature:
            self.df_dim += self.num_bins
            # self.ln_LPS = ChannelWiseLayerNorm(self.num_bins)
    def forward(self, all):
        """
        Compute directional features.
        :param all:
        [0] x - input mixture waveform, with shape [batch_size (B), n_channel (M), seq_len (S)]
        [1] directions - all speakers' directions with shape [batch_size (B), n_spk (C)]
        [2] spk_num - actual speaker number in current wav [batch_size (B)]
        :return: spatial features & directional features, with shape [batch size (B), ?, K]
        """
        # analyzing directional features
        x = all[0]
        #directions = all[1]
        #nspk = all[2]

        batch_size, n_channel, S_ = x.shape
        # B, M, S -> BxM, S
        all_s = x.view(-1, S_)
        # BxM, F, K
        magnitude, phase = self.stft(all_s)
        _, F_, K_ = phase.shape
        # B, M, F, K
        phase = phase.view(batch_size, n_channel, F_, K_)
        magnitude = magnitude.view(batch_size, n_channel, F_, K_)

        df = []
        if 'LPS' in self.input_feature:
            lps = torch.log(magnitude[:, 0] ** 2 + self.epsilon)
            # lps = self.ln_LPS(lps)
            df.append(lps)
        df = torch.cat(df, dim=1)

        return df, magnitude, phase