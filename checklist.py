import torch
from filterbanks.enc_dec import Encoder, Decoder
from filterbanks.stft_fb import STFTFB
from filterbanks.analytic_free_fb import AnalyticFreeFB
from pystoi import stoi
import numpy as np

import torch.fft
from scipy import signal

if __name__ == "__main__": 
    kernel_size = 16
    a = np.hanning(kernel_size + 1)[:-1]
    b = np.hanning(kernel_size)

    ref = torch.range(1, 16000)
    test = torch.fft.fft(ref)
    test = torch.fft.ifft(test)
    f, t, Zxx = signal.stft(ref, fs=16000)
    _, test2 = signal.istft(Zxx, 16000)
    result0 = torch.sum((ref - test2) ** 2)
    
    stft = STFTFB(kernel_size=16, n_filters=16, stride=8, sample_rate = 16000)
    encoder = Encoder(STFTFB(kernel_size=320, n_filters=320, stride=160, sample_rate = 16000))
    decoder = Decoder(STFTFB(kernel_size=320, n_filters=320, stride=160, sample_rate = 16000))
    wav = decoder(encoder(ref))
    result1 = torch.sum((ref - wav) ** 2)

    free = AnalyticFreeFB(kernel_size=320, n_filters=320, stride=160, sample_rate = 16000)
    encoder = Encoder(AnalyticFreeFB(kernel_size=320, n_filters=320, stride=160, sample_rate = 16000))
    decoder = Decoder(AnalyticFreeFB(kernel_size=320, n_filters=320, stride=160, sample_rate = 16000))
    wav = decoder(encoder(ref))
    result2 = torch.sum((ref - wav) ** 2)
    
    a = stoi(ref[0], ref[0], 16000)
    print("Done")