import math
import torch
from typing import Tuple

from .scripting import script_if_tracing
from .deprecation import mark_deprecated


def mul_c(inp, other, dim: int = -2):
    """Entrywise product for complex valued tensors.
    Operands are assumed to have the real parts of each entry followed by the
    imaginary parts of each entry along dimension `dim`, e.g. for,
    ``dim = 1``, the matrix
    .. code::
        [[1, 2, 3, 4],
         [5, 6, 7, 8]]
    is interpreted as
    .. code::
        [[1 + 3j, 2 + 4j],
         [5 + 7j, 6 + 8j]
    where `j` is such that `j * j = -1`.
    Args:
        inp (:class:`torch.Tensor`): The first operand with real and
            imaginary parts concatenated on the `dim` axis.
        other (:class:`torch.Tensor`): The second operand.
        dim (int, optional): frequency (or equivalent) dimension along which
            real and imaginary values are concatenated.
    Returns:
        :class:`torch.Tensor`:
            The complex multiplication between `inp` and `other`
            For now, it assumes that `other` has the same shape as `inp` along
            `dim`.
    """
    check_complex(inp, dim=dim)
    check_complex(other, dim=dim)
    real1, imag1 = inp.chunk(2, dim=dim)
    real2, imag2 = other.chunk(2, dim=dim)
    return torch.cat([real1 * real2 - imag1 * imag2, real1 * imag2 + imag1 * real2], dim=dim)


def reim(x, dim: int = -2) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns a tuple (re, im).
    Args:
        x (:class:`torch.Tensor`): Complex valued tensor.
        dim (int): frequency (or equivalent) dimension along which real and
            imaginary values are concatenated.
    """
    return torch.chunk(x, 2, dim=dim)


def mag(x, dim: int = -2, EPS: float = 1e-8):
    """Takes the magnitude of a complex tensor.
    The operands is assumed to have the real parts of each entry followed by
    the imaginary parts of each entry along dimension `dim`, e.g. for,
    ``dim = 1``, the matrix
    .. code::
        [[1, 2, 3, 4],
         [5, 6, 7, 8]]
    is interpreted as
    .. code::
        [[1 + 3j, 2 + 4j],
         [5 + 7j, 6 + 8j]
    where `j` is such that `j * j = -1`.
    Args:
        x (:class:`torch.Tensor`): Complex valued tensor.
        dim (int): frequency (or equivalent) dimension along which real and
            imaginary values are concatenated.
    Returns:
        :class:`torch.Tensor`: The magnitude of x.
    """
    check_complex(x, dim=dim)
    power = torch.stack(torch.chunk(x, 2, dim=dim), dim=-1).pow(2).sum(dim=-1)
    power = power + EPS
    return power.pow(0.5)


def magreim(x, dim: int = -2):
    """Returns a concatenation of (mag, re, im).
    Args:
        x (:class:`torch.Tensor`): Complex valued tensor.
        dim (int): frequency (or equivalent) dimension along which real and
            imaginary values are concatenated.
    """
    return torch.cat([mag(x, dim=dim), x], dim=dim)


def apply_real_mask(tf_rep, mask, dim: int = -2):
    """Applies a real-valued mask to a real-valued representation.
    It corresponds to ReIm mask in [1].
    Args:
        tf_rep (:class:`torch.Tensor`): The time frequency representation to
            apply the mask to.
        mask (:class:`torch.Tensor`): The real-valued mask to be applied.
        dim (int): Kept to have the same interface with the other ones.
    Returns:
        :class:`torch.Tensor`: `tf_rep` multiplied by the `mask`.
    """
    return tf_rep * mask


def apply_mag_mask(tf_rep, mask, dim: int = -2):
    """Applies a real-valued mask to a complex-valued representation.
    If `tf_rep` has 2N elements along `dim`, `mask` has N elements, `mask` is
    duplicated along `dim` to apply the same mask to both the Re and Im.
    `tf_rep` is assumed to have the real parts of each entry followed by
    the imaginary parts of each entry along dimension `dim`, e.g. for,
    ``dim = 1``, the matrix
    .. code::
        [[1, 2, 3, 4],
         [5, 6, 7, 8]]
    is interpreted as
    .. code::
        [[1 + 3j, 2 + 4j],
         [5 + 7j, 6 + 8j]
    where `j` is such that `j * j = -1`.
    Args:
        tf_rep (:class:`torch.Tensor`): The time frequency representation to
            apply the mask to. Re and Im are concatenated along `dim`.
        mask (:class:`torch.Tensor`): The real-valued mask to be applied.
        dim (int): The frequency (or equivalent) dimension of both `tf_rep` and
            `mask` along which real and imaginary values are concatenated.
    Returns:
        :class:`torch.Tensor`: `tf_rep` multiplied by the `mask`.
    """
    check_complex(tf_rep, dim=dim)
    mask = torch.cat([mask, mask], dim=dim)
    return tf_rep * mask


def apply_complex_mask(tf_rep, mask, dim: int = -2):
    """Applies a complex-valued mask to a complex-valued representation.
    Operands are assumed to have the real parts of each entry followed by the
    imaginary parts of each entry along dimension `dim`, e.g. for,
    ``dim = 1``, the matrix
    .. code::
        [[1, 2, 3, 4],
         [5, 6, 7, 8]]
    is interpreted as
    .. code::
        [[1 + 3j, 2 + 4j],
         [5 + 7j, 6 + 8j]
    where `j` is such that `j * j = -1`.
    Args:
        tf_rep (:class:`torch.Tensor`): The time frequency representation to
            apply the mask to.
        mask (class:`torch.Tensor`): The complex-valued mask to be applied.
        dim (int): The frequency (or equivalent) dimension of both `tf_rep` an
            `mask` along which real and imaginary values are concatenated.
    Returns:
        :class:`torch.Tensor`:
            `tf_rep` multiplied by the `mask` in the complex sense.
    """
    check_complex(tf_rep, dim=dim)
    return mul_c(tf_rep, mask, dim=dim)


def is_asteroid_complex(tensor, dim: int = -2):
    """Check if tensor is complex-like in a given dimension.
    Args:
        tensor (torch.Tensor): tensor to be checked.
        dim(int): the frequency (or equivalent) dimension along which
            real and imaginary values are concatenated.
    Returns:
        True if dimension is even in the specified dimension, otherwise False
    """
    return tensor.shape[dim] % 2 == 0


def check_complex(tensor, dim: int = -2):
    """Assert that tensor is an Asteroid-style complex in a given dimension.
    Args:
        tensor (torch.Tensor): tensor to be checked.
        dim(int): the frequency (or equivalent) dimension along which
            real and imaginary values are concatenated.
    Raises:
        AssertionError if dimension is not even in the specified dimension
    """
    if not is_asteroid_complex(tensor, dim):
        raise AssertionError(
            f"Could not equally chunk the tensor (shape {tensor.shape}) "
            f"along the given dimension ({dim}). Dim axis is "
            "probably wrong"
        )


def to_numpy(tensor, dim: int = -2):
    """Convert complex-like torch tensor to numpy complex array
    Args:
        tensor (torch.Tensor): Complex tensor to convert to numpy.
        dim(int, optional): the frequency (or equivalent) dimension along which
            real and imaginary values are concatenated.
    Returns:
        :class:`numpy.array`:
            Corresponding complex array.
    """
    check_complex(tensor, dim=dim)
    real, imag = torch.chunk(tensor, 2, dim=dim)
    return real.data.numpy() + 1j * imag.data.numpy()


def from_numpy(array, dim: int = -2):
    """Convert complex numpy array to complex-like torch tensor.
    Args:
        array (np.array): array to be converted.
        dim(int, optional): the frequency (or equivalent) dimension along which
            real and imaginary values are concatenated.
    Returns:
        :class:`torch.Tensor`:
            Corresponding torch.Tensor (complex axis in dim `dim`=
    """
    import numpy as np  # Hub-importable

    return torch.cat([torch.from_numpy(np.real(array)), torch.from_numpy(np.imag(array))], dim=dim)


@script_if_tracing
def is_torchaudio_complex(x):
    """Check if tensor is Torchaudio-style complex-like (last dimension is 2).
    Args:
        x (torch.Tensor): tensor to be checked.
    Returns:
        True if last dimension is 2, else False.
    """
    return x.shape[-1] == 2


@script_if_tracing
def check_torchaudio_complex(tensor):
    """Assert that tensor is Torchaudo-style complex-like (last dimension is 2).
    Args:
        tensor (torch.Tensor): tensor to be checked.
    Raises:
        AssertionError if last dimension is != 2.
    """
    if not is_torchaudio_complex(tensor):
        raise AssertionError(
            f"Tensor of shape {tensor.shape} is not Torchaudio-style complex-like"
            "(expected last dimension to be == 2)"
        )


def to_torchaudio(tensor, dim: int = -2):
    """Converts complex-like torch tensor to torchaudio style complex tensor.
    Args:
        tensor (torch.tensor): asteroid-style complex-like torch tensor.
        dim(int, optional): the frequency (or equivalent) dimension along which
            real and imaginary values are concatenated.
    Returns:
        :class:`torch.Tensor`:
            torchaudio-style complex-like torch tensor.
    """
    return torch.stack(torch.chunk(tensor, 2, dim=dim), dim=-1)


@script_if_tracing
def from_torchaudio(tensor, dim: int = -2):
    """Converts torchaudio style complex tensor to complex-like torch tensor.
    Args:
        tensor (torch.tensor): torchaudio-style complex-like torch tensor.
        dim(int, optional): the frequency (or equivalent) dimension along which
            real and imaginary values are concatenated.
    Returns:
        :class:`torch.Tensor`:
            asteroid-style complex-like torch tensor.
    """
    dim = dim - 1 if dim < 0 else dim
    return torch.cat(torch.chunk(tensor, 2, dim=-1), dim=dim).squeeze(-1)


def to_torch_complex(tensor, dim: int = -2):
    """Converts complex-like torch tensor to native PyTorch complex tensor.
    Args:
        tensor (torch.tensor): asteroid-style complex-like torch tensor.
        dim(int, optional): the frequency (or equivalent) dimension along which
            real and imaginary values are concatenated.
    Returns:
        :class:`torch.Tensor`:
            Pytorch native complex-like torch tensor.
    """
    return torch.view_as_complex(to_torchaudio(tensor, dim=dim))


def from_torch_complex(tensor, dim: int = -2):
    """Converts Pytorch native complex tensor to complex-like torch tensor.
    Args:
        tensor (torch.tensor): PyTorch native complex-like torch tensor.
        dim(int, optional): the frequency (or equivalent) dimension along which
            real and imaginary values are concatenated.
    Returns:
        :class:`torch.Tensor`:
            asteroid-style complex-like torch tensor.
    """
    return torch.cat([tensor.real, tensor.imag], dim=dim)


def angle(tensor, dim: int = -2):
    """Return the angle of the complex-like torch tensor.
    Args:
        tensor (torch.Tensor): the complex tensor from which to extract the
            phase.
        dim(int, optional): the frequency (or equivalent) dimension along which
            real and imaginary values are concatenated.
    Returns:
        :class:`torch.Tensor`:
            The counterclockwise angle from the positive real axis on
            the complex plane in radians.
    """
    check_complex(tensor, dim=dim)
    real, imag = torch.chunk(tensor, 2, dim=dim)
    return torch.atan2(imag, real)


def from_magphase(mag_spec, phase, dim: int = -2):
    """Return a complex-like torch tensor from magnitude and phase components.
    Args:
        mag_spec (torch.tensor): magnitude of the tensor.
        phase (torch.tensor): angle of the tensor
        dim(int, optional): the frequency (or equivalent) dimension along which
            real and imaginary values are concatenated.
    Returns:
        :class:`torch.Tensor`:
            The corresponding complex-like torch tensor.
    """
    return torch.cat([mag_spec * torch.cos(phase), mag_spec * torch.sin(phase)], dim=dim)


def magphase(spec: torch.Tensor, dim: int = -2) -> Tuple[torch.Tensor, torch.Tensor]:
    """Splits Asteroid complex-like tensor into magnitude and phase."""
    mag_val = mag(spec, dim=dim)
    phase = angle(spec, dim=dim)
    return mag_val, phase


def centerfreq_correction(
    spec: torch.Tensor,
    kernel_size: int,
    stride: int = None,
    dim: int = -2,
) -> torch.Tensor:
    """Corrects phase from the input spectrogram so that a sinusoid in the
    middle of a bin keeps the same phase from one frame to the next.
    Args:
        spec: Spectrogram tensor of shape (batch, n_freq + 2, frames).
        kernel_size (int): Kernel size of the STFT.
        stride (int): Stride of the STFT.
        dim (int): Only works of dim=-2.
    Returns:
        Tensor: the input spec with corrected phase.
    """
    if dim != -2:
        raise NotImplementedError
    if stride is None:
        stride = kernel_size // 2
    # Phase will be (batch, n_freq // 2 + 1, frames)
    mag_spec, phase = magphase(spec, dim=dim)
    new_phase = phase_centerfreq_correction(phase, kernel_size=kernel_size, stride=stride)
    new_spec = from_magphase(mag_spec, new_phase, dim=dim)
    return new_spec


def phase_centerfreq_correction(
    phase: torch.Tensor,
    kernel_size: int,
    stride: int = None,
) -> torch.Tensor:
    """Corrects phase so that a sinusoid in the middle of a bin keeps the
    same phase from one frame to the next.
    Args:
        phase: tensor of shape (batch, n_freq//2 + 1, frames)
        kernel_size (int): Kernel size of the STFT.
        stride (int): Stride of the STFT.
    Returns:
        Tensor: corrected phase.
    """
    *_, freq, frames = phase.shape
    tmp = torch.arange(freq).unsqueeze(-1) * torch.arange(frames)[None]
    correction = -2 * tmp * stride * math.pi / kernel_size
    return phase + correction

