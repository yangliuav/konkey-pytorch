import torch
from filterbanks.enc_dec import Encoder, Decoder
from filterbanks.stft_fb import STFTFB
from filterbanks.torch_stft_fb import TorchSTFTFB
# from filterbanks.analytic_free_fb import AnalyticFreeFB
from filterbanks.stft_df import STFT, iSTFT
from pystoi import stoi
import numpy as np

import torch.fft
from scipy import signal
# from asteroid.filterbanks import Encoder, Decoder
# from asteroid.filterbanks import STFTFB # FreeFB, AnalyticFreeFB, ParamSincFB, MultiphaseGammatoneFB

if __name__ == "__main__": 
    ref = torch.range(1, 16000)
    test1 = torch.stft(ref,  n_fft = 320)
    test = torch.istft(test1,  n_fft = 320)
    result0 = torch.sum((ref - test) ** 2)

    f, t, Zxx = signal.stft(ref, fs=16000)
    _, test2 = signal.istft(Zxx, 16000)
    result1 = torch.sum((ref - test2) ** 2)
    
    encoder = Encoder(TorchSTFTFB(kernel_size=320, n_filters=320, sample_rate = 16000))
    decoder = Decoder(TorchSTFTFB(kernel_size=320, n_filters=320, sample_rate = 16000))
    wav = decoder(encoder(ref))
    result2 = torch.sum((ref - wav[:16000]) ** 2)

    stft = STFT(frame_len= 320, frame_hop=160)
    istft = iSTFT(frame_len=320, frame_hop=160)
    test3 = istft(stft(ref))
    

    # free = AnalyticFreeFB(kernel_size=320, n_filters=320, stride=160, sample_rate = 16000)
    # encoder = Encoder(AnalyticFreeFB(kernel_size=320, n_filters=320, stride=160, sample_rate = 16000))
    # decoder = Decoder(AnalyticFreeFB(kernel_size=320, n_filters=320, stride=160, sample_rate = 16000))
    # wav = decoder(encoder(ref))
    # result2 = torch.sum((ref - wav) ** 2)
    
    a = stoi(ref[0], ref[0], 16000)
    print("Done")