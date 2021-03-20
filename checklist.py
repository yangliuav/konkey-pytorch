import torch
from filterbanks.enc_dec import Encoder, Decoder
from filterbanks.stft_fb import STFTFB
from filterbanks.analytic_free_fb import AnalyticFreeFB

if __name__ == "__main__": 
    ref = torch.randn(1, 16000)
    encoder = Encoder(STFTFB(kernel_size=320, n_filters=320, stride=160, sample_rate = 16000))
    decoder = Decoder(STFTFB(kernel_size=320, n_filters=320, stride=160, sample_rate = 16000))
    wav = decoder(encoder(ref))
    result1 = torch.sum((ref - wav) ** 2)

    encoder = Encoder(AnalyticFreeFB(kernel_size=320, n_filters=320, stride=160, sample_rate = 16000))
    decoder = Decoder(AnalyticFreeFB(kernel_size=320, n_filters=320, stride=160, sample_rate = 16000))
    wav = decoder(encoder(ref))
    result2 = torch.sum((ref - wav) ** 2)
    print("Done")