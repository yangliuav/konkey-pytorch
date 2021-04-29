import torch

class SpecAugment(torch.nn.Module):
    """ 
    An implementation of SpecAugment algorithm.

    Reference:
        https://arxiv.org/abs/1904.08779

    Arguments
    ---------
    time_warp : bool
        Whether applying time warping.
    time_warp_window : int
        Time warp window.
    time_warp_mode : str
        Interpolation mode for time warping (default "bicubic").
    freq_mask : bool1
        Whether applying freq mask.
    freq_mask_width : int or tuple
        Freq mask width range.
    n_freq_mask : int
        Number of freq mask.
    time_mask : int
        Whether applying time mask.
    time_mask_width : int or tuple
        Time mask width range.
    n_time_mask : int
        Number of time mask.
    replace_with_zero : bool
        If True, replace masked value with 0, else replace masked value with mean of the input tensor.

    Example
    -------
    >>> aug = SpecAugment()
    >>> a = torch.rand([8, 120, 80])
    >>> a = aug(a)
    >>> print(a.shape)
    torch.Size([8, 120, 80])"

    Note
    -------
    (Batch, Freq, Time)
    """

    def __init__(
        self,
        time_warp = True,
        time_warp_window = 5,
        time_warp_mode = "bicubic",
        freq_mask = True,
        freq_mask_width = (0, 20),
        n_freq_mask = 2, 
        time_mask = True,
        time_mask_width = (0, 100),
        n_time_mask = 2,
        replace_with_zero = True,
    ):
        super().__init__()
        assert(
            time_warp or freq_mask or time_warp
        ), "at least one of time_warp, time_mask, or freq_mask should be applied"

        # Time Warp Setting
        self.time_warp = time_warp
        self.time_warp_window = time_warp_window
        self.time_warp_mode = time_warp_mode 
        
        # Freq Mask Setting
        self.freq_mask = freq_mask
        if isinstance(freq_mask_width, int):
            freq_mask_width = (0, freq_mask_width)
        self.freq_mask_width = freq_mask_width
        self.n_freq_mask = n_freq_mask

        # Time Mask Setting
        self.time_mask = time_mask
        if isinstance(time_mask_width, int):
            time_mask_width = (0, time_warp_window)
        self.time_mask_width = time_mask_width
        self.n_time_mask = n_time_mask

        self.replace_with_zero = replace_with_zero

    def forward(self, x):
        if self.time_warp:
            x = self.time_warp(x)
        if self.freq_mask:
            x = self.mask_along_axis(x, dim = 1)
        if self.time_mask:
            x = self.mask_along_axis(x, dim = 2)
        return x
    
    def time_warp(self, x):
        """Time warping with torch.nn.functional.interpolate"""
        
        original_size = x.shape
        window = self.time_warp_window

        time = x.shape[2]
        if time - window <= window:
            return x

        # 2d interpolation requires 4D or higher dimension tensors
        # x: (Batch, Freq, Time) -> (Batch, 1, Freq, Time)
        if x.dim() == 3:
            x = x.unsqueeze(1)

        # compute center and corepoding window
        center = torch.randint(window, time - window, (1,))[0]
        w = torch.randint(center - window, center + window, (1,))[0] + 1

        left = torch.nn.functional.interpolate(
            x[:, :, :, :center],
            (x.shape[2], w), 
            mode=self.time_warp_mode,
            align_corners=True,
        )

        right = torch.nn.functional.interpolate(
            x[:, :, :, center:],
            (x.shape[2], time - w),
            mode=self.time_warp_mode,
            align_corners=True,
        )  

        x[:, :, :, :w] = left
        x[:, :, :, w:] = right 

        return x.view(*original_size)

    def mask_along_axis(self, x, dim):
        """
        Mask along time or frequency axis.

        Arguments
        ---------
        x : tensor
            Input tensor
        dim : int
            Corresponding dimension to mask.
        """
        original_size = x.shape 
        if x.shape == 4:
            x = x.view(-1, x.shape[2], x.shape[3])

        batch, fea, time = x.shape 

        if dim == 2:
            D = time
            n_mask = self.n_time_mask
            width_range = self.time_mask_width
        else:
            D = fea 
            n_mask = self.n_freq_mask
            width_range = self.freq_mask_width

        mask_len = torch.randint(
            width_range[0], width_range[1], (batch, n_mask), device=x.device 
        ).unsqueeze(2)

        mask_pos = torch.randint(
            0, max(1, D - mask_len.max()), (batch, n_mask), device=x.device 
        ).unsqueeze(2)

        # compute masks
        arange = torch.arange(D, device = x.device).view(1, 1, -1)
        mask = (mask_pos <= arange) * (arange < (mask_pos + mask_len))

        if dim == 2:
            mask = mask.unsqueeze(2)
        else:
            mask = mask.unsqueeze(1)
        
        if self.replace_with_zero:
            val = 0.0
        else:
            val = x.mean()

        x = x.masked_fill_(mask, val)
        return x.view(*original_size)


