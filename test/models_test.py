import torch
import pytest
from torch.testing import assert_allclose
import numpy as np
import soundfile as sf
from models.dcunet import DCUNet
from filterbanks import make_enc_dec

def _default_test_model(model, input_samples=801, test_input=None):
    if test_input is None:
        test_input = torch.randn(1, input_samples)

    model_conf = model.serialize()
    reconstructed_model = model.__class__.from_pretrained(model_conf)
    assert_allclose(model(test_input), reconstructed_model(test_input))

    # Load with and without SR
    sr = model_conf["model_args"].pop("sample_rate")
    reconstructed_model_nosr = model.__class__.from_pretrained(model_conf)
    reconstructed_model = model.__class__.from_pretrained(model_conf, sample_rate=sr)

    assert reconstructed_model.sample_rate == model.sample_rate


def test_dcunet():
    n_fft = 1024
    _, istft = make_enc_dec(
        "stft", n_filters=n_fft, kernel_size=1024, stride=256, sample_rate=16000
    )
    input_samples = istft(torch.zeros((n_fft + 2, 17))).shape[0]
    _default_test_model(DCUNet("DCUNet-10"), input_samples=input_samples)
    _default_test_model(DCUNet("DCUNet-16"), input_samples=input_samples)
    _default_test_model(DCUNet("DCUNet-20"), input_samples=input_samples)
    _default_test_model(DCUNet("Large-DCUNet-20"), input_samples=input_samples)
    _default_test_model(DCUNet("DCUNet-10", n_src=2), input_samples=input_samples)

    # DCUMaskNet should fail with wrong freqency dimensions
    DCUNet("mini").masker(torch.zeros((1, 9, 17), dtype=torch.complex64))
    with pytest.raises(TypeError):
        DCUNet("mini").masker(torch.zeros((1, 42, 17), dtype=torch.complex64))

    # DCUMaskNet should fail with wrong time dimensions if fix_length_mode is not used
    DCUNet("mini", fix_length_mode="pad").masker(torch.zeros((1, 9, 17), dtype=torch.complex64))
    DCUNet("mini", fix_length_mode="trim").masker(torch.zeros((1, 9, 17), dtype=torch.complex64))
    with pytest.raises(TypeError):
        DCUNet("mini").masker(torch.zeros((1, 9, 16), dtype=torch.complex64))

test_dcunet()