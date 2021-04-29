import torch
from dsp.resample import Resample  

class SpeedPerturb(torch.nn.Module):
    """Slightly speed up or slow down an audio signal.
    Resample the audio signal at a rate that is similar to the original rate,
    to achieve a slightly slower or slightly faster signal. This technique is
    outlined in the paper: "Audio Augmentation for Speech Recognition"
    Arguments
    ---------
    orig_freq : int
        The frequency of the original signal.
    speeds : list
        The speeds that the signal should be changed to, as a percentage of the
        original signal (i.e. `speeds` is divided by 100 to get a ratio).
    perturb_prob : float
        The chance that the batch will be speed-
        perturbed. By default, every batch is perturbed.
    Example
    -------
    >>> from speechbrain.dataio.dataio import read_audio
    >>> signal = read_audio('samples/audio_samples/example1.wav')
    >>> perturbator = SpeedPerturb(orig_freq=16000, speeds=[90])
    >>> clean = signal.unsqueeze(0)
    >>> perturbed = perturbator(clean)
    >>> clean.shape
    torch.Size([1, 52173])
    >>> perturbed.shape
    torch.Size([1, 46956])
    """

    def __init__(
        self, orig_freq, speeds=[90, 100, 110], perturb_prob=1.0,
    ):
        super().__init__()
        self.orig_freq = orig_freq
        self.speeds = speeds
        self.perturb_prob = perturb_prob

        # Initialize index of perturbation
        self.samp_index = 0

        # Initialize resamplers
        self.resamplers = []
        for speed in self.speeds:
            config = {
                "orig_freq": self.orig_freq,
                "new_freq": self.orig_freq * speed // 100,
            }
            self.resamplers.append(Resample(**config))

    def forward(self, waveform):
        """
        Arguments
        ---------
        waveforms : tensor
            Shape should be `[batch, time]` or `[batch, channels, time]`.
        lengths : tensor
            Shape should be a single dimension, `[batch]`.
        Returns
        -------
        Tensor of shape `[batch, time]` or `[batch, channels, time]`.
        """

        # Don't perturb (return early) 1-`perturb_prob` portion of the batches
        if torch.rand(1) > self.perturb_prob:
            return waveform.clone()

        # Perform a random perturbation
        self.samp_index = torch.randint(len(self.speeds), (1,))[0]
        perturbed_waveform = self.resamplers[self.samp_index](waveform)

        return perturbed_waveform