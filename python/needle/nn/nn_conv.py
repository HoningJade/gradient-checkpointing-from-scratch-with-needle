"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
# , Sequential, BatchNorm2d, Residual, Flatten, Linear, ReLU
from .nn_basic import Parameter, Module


class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        # BEGIN YOUR SOLUTION
        self.padding = self.kernel_size // 2
        # kernel
        fan_in = self.in_channels * self.kernel_size ** 2
        fan_out = self.out_channels * self.kernel_size ** 2
        shape = (self.kernel_size, self.kernel_size,
                 self.in_channels, self.out_channels)
        weight = init.kaiming_uniform(fan_in, fan_out, shape=shape)
        self.weight = Parameter(weight, device=device, dtype=dtype)
        # bias
        self.use_bias = bias
        if self.use_bias:
            k_val = 1.0 / (fan_in ** 0.5)
            b = init.rand(self.out_channels, low=-k_val, high=k_val)
            self.bias = Parameter(b, device=device, dtype=dtype)
        # END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        # BEGIN YOUR SOLUTION
        # NCHW -> NHWC
        x = ops.transpose(x, (2, 1))
        x = ops.transpose(x, (3, 2))
        output = ops.conv(x, self.weight, self.stride, self.padding)
        # 1C -> 111C -> NHWC
        if self.use_bias:
            bias = ops.reshape(self.bias, (1, 1, 1, self.out_channels))
            bias = ops.broadcast_to(bias, output.shape)
            output += bias
        # NHWC-> NCHW
        output = ops.transpose(output, (3, 2))
        output = ops.transpose(output, (2, 1))
        return output
        # END YOUR SOLUTION
