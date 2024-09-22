import torch.nn as nn 
import torch
import torch.nn.functional as F

#copied from https://github.com/mazzzystar/WaveGAN-pytorch/blob/master/wavegan.py
#that copied from https://github.com/jtcramer/wavegan/blob/master/wavegan.py#L8
class PhaseShuffle(nn.Module):
    def __init__(self, shift_factor):
        super(PhaseShuffle, self).__init__()
        self.shift_factor = shift_factor

    def forward(self, x):
        if self.shift_factor == 0:
            return x
        # uniform in (L, R)
        k_list = torch.Tensor(x.shape[0]).random_(0, 2 * self.shift_factor + 1) - self.shift_factor
        k_list = k_list.numpy().astype(int)

        # Combine sample indices into lists so that less shuffle operations
        # need to be performed
        k_map = {}
        for idx, k in enumerate(k_list):
            k = int(k)
            if k not in k_map:
                k_map[k] = []
            k_map[k].append(idx)

        # Make a copy of x for our output
        x_shuffle = x.clone()

        # Apply shuffle to each sample
        for k, idxs in k_map.items():
            if k > 0:
                x_shuffle[idxs] = F.pad(x[idxs][..., :-k], (k, 0), mode='reflect')
            else:
                x_shuffle[idxs] = F.pad(x[idxs][..., -k:], (0, -k), mode='reflect')

        assert x_shuffle.shape == x.shape, "{}, {}".format(x_shuffle.shape,
                                                       x.shape)
        return x_shuffle

#from https://github.com/mostafaelaraby/wavegan-pytorch/blob/master/models.py
class Transposed1DConv(nn.Module):
    def __init__(self, in_channels,
            out_channels,
            kernel_size,
            stride,
            padding=11,
            upsample=None,
            output_padding=1,
            use_batch_norm=False,
            const_pad = None):
        super(Transposed1DConv, self).__init__()

        if const_pad is None:
            const_pad = kernel_size // 2

        self.upsample = upsample
        reflection_pad = nn.ConstantPad1d(const_pad, value=0)
        conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride)
        conv1d.weight.data.normal_(0.0, 0.02)
        Conv1dTrans = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride, padding, output_padding
        )
        batch_norm = nn.BatchNorm1d(out_channels)
        if self.upsample:
            operation_list = [reflection_pad, conv1d]
        else:
            operation_list = [Conv1dTrans]

        if use_batch_norm:
            operation_list.append(batch_norm)
        self.transpose_ops = nn.Sequential(*operation_list)

    def forward(self, x):
        if self.upsample:
            # recommended by wavgan paper to use nearest upsampling
            x = nn.functional.interpolate(x, scale_factor=self.upsample, mode="nearest")
        return self.transpose_ops(x)