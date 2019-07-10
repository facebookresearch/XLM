import torch
from torch import nn

from .utils import get_slices


def mlp(sizes, bias=True, batchnorm=True, groups=1):
    """
    Generate a feedforward neural network.
    """
    assert len(sizes) >= 2
    pairs = [(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)]
    layers = []

    for i, (dim_in, dim_out) in enumerate(pairs):
        if groups == 1 or i == 0:
            layers.append(nn.Linear(dim_in, groups * dim_out, bias=bias))
        else:
            layers.append(GroupedLinear(groups * dim_in, groups * dim_out, bias=bias, groups=groups))
        if batchnorm:
            layers.append(nn.BatchNorm1d(groups * dim_out))
        if i < len(pairs) - 1:
            layers.append(nn.ReLU())

    return nn.Sequential(*layers)


def convs(channel_sizes, kernel_sizes, bias=True, batchnorm=True, residual=False, groups=1):
    """
    Generate a convolutional neural network.
    """
    assert len(channel_sizes) >= 2
    assert len(channel_sizes) == len(kernel_sizes) + 1
    pairs = [(channel_sizes[i], channel_sizes[i + 1]) for i in range(len(channel_sizes) - 1)]
    layers = []

    for i, (dim_in, dim_out) in enumerate(pairs):
        ks = (kernel_sizes[i], kernel_sizes[i])
        in_group = 1 if i == 0 else groups
        _dim_in = dim_in * in_group
        _dim_out = dim_out * groups
        if not residual:
            layers.append(nn.Conv2d(_dim_in, _dim_out, ks, padding=[k // 2 for k in ks], bias=bias, groups=in_group))
            if batchnorm:
                layers.append(nn.BatchNorm2d(_dim_out))
            if i < len(pairs) - 1:
                layers.append(nn.ReLU())
        else:
            layers.append(BottleneckResidualConv2d(
                _dim_in, _dim_out, ks, bias=bias,
                batchnorm=batchnorm, groups=in_group
            ))
            if i == len(pairs) - 1:
                layers.append(nn.Conv2d(_dim_out, _dim_out, (1, 1), bias=bias))

    return nn.Sequential(*layers)


class GroupedLinear(nn.Module):

    def __init__(self, in_features, out_features, bias=True, groups=1):

        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.groups = groups
        self.bias = bias
        assert groups > 1

        self.layer = nn.Conv1d(in_features, out_features, bias=bias, kernel_size=1, groups=groups)

    def forward(self, input):
        assert input.dim() == 2 and input.size(1) == self.in_features
        return self.layer(input.unsqueeze(2)).squeeze(2)

    def extra_repr(self):
        return 'in_features={}, out_features={}, groups={}, bias={}'.format(
            self.in_features, self.out_features, self.groups, self.bias is not None
        )


class BottleneckResidualConv2d(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size, bias=True, batchnorm=True, groups=1):

        super().__init__()
        hidden_channels = min(input_channels, output_channels)
        assert all(k % 2 == 1 for k in kernel_size)

        self.conv1 = nn.Conv2d(input_channels, hidden_channels, kernel_size, padding=[k // 2 for k in kernel_size], bias=bias, groups=groups)
        self.conv2 = nn.Conv2d(hidden_channels, output_channels, kernel_size, padding=[k // 2 for k in kernel_size], bias=bias, groups=groups)
        self.act = nn.ReLU()

        self.batchnorm = batchnorm
        if self.batchnorm:
            self.bn1 = nn.BatchNorm2d(hidden_channels)
            self.bn2 = nn.BatchNorm2d(output_channels)

        if input_channels == output_channels:
            self.residual = nn.Sequential()
        else:
            self.residual = nn.Conv2d(input_channels, output_channels, (1, 1), bias=False, groups=groups)

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x) if self.batchnorm else x
        x = self.act(x)
        x = self.conv2(x)
        x = self.bn2(x) if self.batchnorm else x
        x = self.act(x + self.residual(input))
        return x


class QueryIdentity(nn.Module):

    def __init__(self, input_dim, heads, shuffle_hidden):
        super().__init__()
        self.input_dim = input_dim
        self.heads = heads
        self.shuffle_query = shuffle_hidden
        assert shuffle_hidden is False or heads > 1
        assert shuffle_hidden is False or self.input_dim % (2 ** self.heads) == 0
        if shuffle_hidden:
            self.slices = {head_id: get_slices(input_dim, head_id) for head_id in range(heads)}

    def forward(self, input):
        """
        Generate queries from hidden states by either
        repeating them or creating some shuffled version.
        """
        assert input.shape[-1] == self.input_dim
        input = input.contiguous().view(-1, self.input_dim) if input.dim() > 2 else input
        bs = len(input)

        if self.heads == 1:
            query = input

        elif not self.shuffle_query:
            query = input.unsqueeze(1).repeat(1, self.heads, 1)
            query = query.view(bs * self.heads, self.input_dim)

        else:
            query = torch.cat([
                input[:, a:b]
                for head_id in range(self.heads)
                for a, b in self.slices[head_id]
            ], 1).view(bs * self.heads, self.input_dim)

        assert query.shape == (bs * self.heads, self.input_dim)
        return query


class QueryMLP(nn.Module):

    def __init__(
        self, input_dim, heads, k_dim, product_quantization, multi_query_net,
        sizes, bias=True, batchnorm=True, grouped_conv=False
    ):
        super().__init__()
        self.input_dim = input_dim
        self.heads = heads
        self.k_dim = k_dim
        self.sizes = sizes
        self.grouped_conv = grouped_conv
        assert not multi_query_net or product_quantization or heads >= 2
        assert sizes[0] == input_dim
        assert sizes[-1] == (k_dim // 2) if multi_query_net else (heads * k_dim)
        assert self.grouped_conv is False or len(sizes) > 2

        # number of required MLPs
        self.groups = (2 * heads) if multi_query_net else 1

        # MLPs
        if self.grouped_conv:
            self.query_mlps = mlp(sizes, bias=bias, batchnorm=batchnorm, groups=self.groups)
        elif len(self.sizes) == 2:
            sizes_ = list(sizes)
            sizes_[-1] = sizes_[-1] * self.groups
            self.query_mlps = mlp(sizes_, bias=bias, batchnorm=batchnorm, groups=1)
        else:
            self.query_mlps = nn.ModuleList([
                mlp(sizes, bias=bias, batchnorm=batchnorm, groups=1)
                for _ in range(self.groups)
            ])

    def forward(self, input):
        """
        Compute queries using either grouped 1D convolutions or ModuleList + concat.
        """
        assert input.shape[-1] == self.input_dim
        input = input.contiguous().view(-1, self.input_dim) if input.dim() > 2 else input
        bs = len(input)

        if self.grouped_conv or len(self.sizes) == 2:
            query = self.query_mlps(input)
        else:
            outputs = [m(input) for m in self.query_mlps]
            query = torch.cat(outputs, 1) if len(outputs) > 1 else outputs[0]

        assert query.shape == (bs, self.heads * self.k_dim)
        return query.view(bs * self.heads, self.k_dim)


class QueryConv(nn.Module):

    def __init__(
        self, input_dim, heads, k_dim, product_quantization, multi_query_net,
        sizes, kernel_sizes, bias=True, batchnorm=True,
        residual=False, grouped_conv=False
    ):
        super().__init__()
        self.input_dim = input_dim
        self.heads = heads
        self.k_dim = k_dim
        self.sizes = sizes
        self.grouped_conv = grouped_conv
        assert not multi_query_net or product_quantization or heads >= 2
        assert sizes[0] == input_dim
        assert sizes[-1] == (k_dim // 2) if multi_query_net else (heads * k_dim)
        assert self.grouped_conv is False or len(sizes) > 2
        assert len(sizes) == len(kernel_sizes) + 1 >= 2 and all(ks % 2 == 1 for ks in kernel_sizes)

        # number of required CNNs
        self.groups = (2 * heads) if multi_query_net else 1

        # CNNs
        if self.grouped_conv:
            self.query_convs = convs(sizes, kernel_sizes, bias=bias, batchnorm=batchnorm, residual=residual, groups=self.groups)
        elif len(self.sizes) == 2:
            sizes_ = list(sizes)
            sizes_[-1] = sizes_[-1] * self.groups
            self.query_convs = convs(sizes_, kernel_sizes, bias=bias, batchnorm=batchnorm, residual=residual, groups=1)
        else:
            self.query_convs = nn.ModuleList([
                convs(sizes, kernel_sizes, bias=bias, batchnorm=batchnorm, residual=residual, groups=1)
                for _ in range(self.groups)
            ])

    def forward(self, input):

        bs, nf, h, w = input.shape
        assert nf == self.input_dim

        if self.grouped_conv or len(self.sizes) == 2:
            query = self.query_convs(input)
        else:
            outputs = [m(input) for m in self.query_convs]
            query = torch.cat(outputs, 1) if len(outputs) > 1 else outputs[0]

        assert query.shape == (bs, self.heads * self.k_dim, h, w)
        query = query.transpose(1, 3).contiguous().view(bs * w * h * self.heads, self.k_dim)
        return query
