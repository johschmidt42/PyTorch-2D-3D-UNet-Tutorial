from enum import Enum, IntEnum
from typing import Optional, Union

import torch
from torch import nn


class ActivationFunction(str, Enum):
    RELU: str = "relu"
    LEAKY: str = "leaky"
    ELU: str = "elu"


class NormalizationLayer(str, Enum):
    BATCH: str = "batch"
    INSTANCE: str = "instance"


class Dimensions(IntEnum):
    TWO: int = 2
    THREE: int = 3


class ConvMode(str, Enum):
    SAME: str = "same"
    VALID: str = "valid"


class UpMode(str, Enum):
    TRANSPOSED: str = "transposed"
    NEAREST: str = "nearest"
    LINEAR: str = "linear"
    BILINEAR: str = "bilinear"
    BICUBIC: str = "bicubic"
    TRILINEAR: str = "trilinear"


@torch.jit.script
def autocrop(encoder_layer: torch.Tensor, decoder_layer: torch.Tensor):
    """
    Center-crops the encoder_layer to the size of the decoder_layer,
    so that merging (concatenation) between levels/blocks is possible.
    This is only necessary for input sizes != 2**n for 'same' padding and always required for 'valid' padding.
    """
    if encoder_layer.shape[2:] != decoder_layer.shape[2:]:
        ds = encoder_layer.shape[2:]
        es = decoder_layer.shape[2:]
        assert ds[0] >= es[0]
        assert ds[1] >= es[1]
        if encoder_layer.dim() == 4:  # 2D
            encoder_layer = encoder_layer[
                :,
                :,
                ((ds[0] - es[0]) // 2) : ((ds[0] + es[0]) // 2),
                ((ds[1] - es[1]) // 2) : ((ds[1] + es[1]) // 2),
            ]
        elif encoder_layer.dim() == 5:  # 3D
            assert ds[2] >= es[2]
            encoder_layer = encoder_layer[
                :,
                :,
                ((ds[0] - es[0]) // 2) : ((ds[0] + es[0]) // 2),
                ((ds[1] - es[1]) // 2) : ((ds[1] + es[1]) // 2),
                ((ds[2] - es[2]) // 2) : ((ds[2] + es[2]) // 2),
            ]
    return encoder_layer, decoder_layer


def conv_layer(dim: int) -> Union[nn.Conv2d, nn.Conv3d]:
    conv_layers: dict = {Dimensions.TWO: nn.Conv2d, Dimensions.THREE: nn.Conv3d}
    return conv_layers[dim]


def get_conv_layer(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 1,
    bias: bool = True,
    dim: int = Dimensions.TWO,
) -> Union[nn.Conv2d, nn.Conv3d]:
    layer: Union[nn.Conv2d, nn.Conv3d] = conv_layer(dim=dim)
    return layer(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=bias,
    )


def conv_transpose_layer(dim: int) -> Union[nn.ConvTranspose2d, nn.ConvTranspose3d]:
    conv_transpose_layers: dict = {
        Dimensions.TWO: nn.ConvTranspose2d,
        Dimensions.THREE: nn.ConvTranspose3d,
    }

    return conv_transpose_layers[dim]


def get_up_layer(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 2,
    stride: int = 2,
    dim: int = Dimensions.TWO,
    up_mode: str = UpMode.TRANSPOSED,
) -> Union[Union[nn.ConvTranspose2d, nn.ConvTranspose3d], nn.Upsample]:
    if up_mode == UpMode.TRANSPOSED:
        return conv_transpose_layer(dim=dim)(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
        )
    else:
        return nn.Upsample(scale_factor=2.0, mode=up_mode)


def maxpool_layer(dim: int) -> Union[nn.MaxPool2d, nn.MaxPool3d]:
    maxpool_layers: dict = {
        Dimensions.TWO: nn.MaxPool2d,
        Dimensions.THREE: nn.MaxPool3d,
    }
    return maxpool_layers[dim]


def get_maxpool_layer(
    kernel_size: int = 2, stride: int = 2, padding: int = 0, dim: int = Dimensions.TWO
) -> Union[nn.MaxPool2d, nn.MaxPool3d]:
    layer = maxpool_layer(dim=dim)
    return layer(kernel_size=kernel_size, stride=stride, padding=padding)


def get_activation_layer(activation: str) -> Union[nn.ReLU, nn.LeakyReLU, nn.ELU]:
    activation_functions: dict = {
        ActivationFunction.RELU: nn.ReLU(),
        ActivationFunction.LEAKY: nn.LeakyReLU(negative_slope=0.1),
        ActivationFunction.ELU: nn.ELU(),
    }

    return activation_functions[activation]


def get_normalization_layer(
    normalization: str, num_channels: int, dim: int
) -> Union[
    Union[nn.BatchNorm2d, nn.BatchNorm3d],
    Union[nn.InstanceNorm2d, nn.InstanceNorm3d],
]:
    normalization_layers: dict = {
        Dimensions.TWO: {
            NormalizationLayer.BATCH: nn.BatchNorm2d(num_channels),
            NormalizationLayer.INSTANCE: nn.InstanceNorm2d(num_channels),
        },
        Dimensions.THREE: {
            NormalizationLayer.BATCH: nn.BatchNorm3d(num_channels),
            NormalizationLayer.INSTANCE: nn.InstanceNorm3d(num_channels),
        },
    }

    return normalization_layers[dim][normalization]


class Concatenate(nn.Module):
    def __init__(self):
        super(Concatenate, self).__init__()

    def forward(self, layer_1, layer_2):
        x = torch.cat((layer_1, layer_2), 1)

        return x


class DownBlock(nn.Module):
    """
    A helper Module that performs 2 Convolutions and 1 MaxPool.
    An activation follows each convolution.
    A normalization layer follows each convolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pooling: bool = True,
        activation: str = ActivationFunction.RELU,
        normalization: Optional[str] = None,
        dim: int = Dimensions.TWO,
        conv_mode: str = ConvMode.SAME,
    ):
        super().__init__()

        conv_modes: dict = {ConvMode.SAME: 1, ConvMode.VALID: 0}

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling
        self.normalization = normalization
        self.padding = conv_modes[conv_mode]
        self.dim = dim
        self.activation = activation

        # conv layers
        self.conv1 = get_conv_layer(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            stride=1,
            padding=self.padding,
            bias=True,
            dim=self.dim,
        )
        self.conv2 = get_conv_layer(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            stride=1,
            padding=self.padding,
            bias=True,
            dim=self.dim,
        )

        # pooling layer
        if self.pooling:
            self.pool = get_maxpool_layer(
                kernel_size=2, stride=2, padding=0, dim=self.dim
            )

        # activation layers
        self.act1 = get_activation_layer(activation=self.activation)
        self.act2 = get_activation_layer(activation=self.activation)

        # normalization layers
        if self.normalization:
            self.norm1 = get_normalization_layer(
                normalization=self.normalization,
                num_channels=self.out_channels,
                dim=self.dim,
            )
            self.norm2 = get_normalization_layer(
                normalization=self.normalization,
                num_channels=self.out_channels,
                dim=self.dim,
            )

    def forward(self, x):
        y = self.conv1(x)  # convolution 1
        y = self.act1(y)  # activation 1
        if self.normalization:
            y = self.norm1(y)  # normalization 1
        y = self.conv2(y)  # convolution 2
        y = self.act2(y)  # activation 2
        if self.normalization:
            y = self.norm2(y)  # normalization 2

        before_pooling = y  # save the outputs before the pooling operation
        if self.pooling:
            y = self.pool(y)  # pooling
        return y, before_pooling


class UpBlock(nn.Module):
    """
    A helper Module that performs 2 Convolutions and 1 UpConvolution/Upsample.
    An activation follows each convolution.
    A normalization layer follows each convolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: str = ActivationFunction.RELU,
        normalization: Optional[str] = None,
        dim: int = Dimensions.TWO,
        conv_mode: str = ConvMode.SAME,
        up_mode: str = UpMode.TRANSPOSED,
    ):
        super().__init__()

        conv_modes: dict = {ConvMode.SAME: 1, ConvMode.VALID: 0}

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalization = normalization
        self.padding = conv_modes[conv_mode]
        self.dim = dim
        self.activation = activation

        self.up_mode = up_mode

        # upconvolution/upsample layer
        self.up = get_up_layer(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=2,
            stride=2,
            dim=self.dim,
            up_mode=self.up_mode,
        )

        # conv layers
        self.conv0 = get_conv_layer(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
            dim=self.dim,
        )
        self.conv1 = get_conv_layer(
            in_channels=2 * self.out_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            stride=1,
            padding=self.padding,
            bias=True,
            dim=self.dim,
        )
        self.conv2 = get_conv_layer(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            stride=1,
            padding=self.padding,
            bias=True,
            dim=self.dim,
        )

        # activation layers
        self.act0 = get_activation_layer(self.activation)
        self.act1 = get_activation_layer(self.activation)
        self.act2 = get_activation_layer(self.activation)

        # normalization layers
        if self.normalization:
            self.norm0 = get_normalization_layer(
                normalization=self.normalization,
                num_channels=self.out_channels,
                dim=self.dim,
            )
            self.norm1 = get_normalization_layer(
                normalization=self.normalization,
                num_channels=self.out_channels,
                dim=self.dim,
            )
            self.norm2 = get_normalization_layer(
                normalization=self.normalization,
                num_channels=self.out_channels,
                dim=self.dim,
            )

        # concatenate layer
        self.concat = Concatenate()

    def forward(self, encoder_layer, decoder_layer):
        """
        Forward pass
        encoder_layer: Tensor from the encoder pathway
        decoder_layer: Tensor from the decoder pathway (to be up'd)
        """
        up_layer = self.up(decoder_layer)  # up-convolution/up-sampling
        cropped_encoder_layer, dec_layer = autocrop(encoder_layer, up_layer)  # cropping

        if self.up_mode != UpMode.TRANSPOSED:
            # We need to reduce the channel dimension with a conv layer
            up_layer = self.conv0(up_layer)  # convolution 0
        up_layer = self.act0(up_layer)  # activation 0
        if self.normalization:
            up_layer = self.norm0(up_layer)  # normalization 0

        merged_layer = self.concat(up_layer, cropped_encoder_layer)  # concatenation
        y = self.conv1(merged_layer)  # convolution 1
        y = self.act1(y)  # activation 1
        if self.normalization:
            y = self.norm1(y)  # normalization 1
        y = self.conv2(y)  # convolution 2
        y = self.act2(y)  # acivation 2
        if self.normalization:
            y = self.norm2(y)  # normalization 2
        return y


class UNet(nn.Module):
    """
    activation: 'relu', 'leaky', 'elu'
    normalization: 'batch', 'instance', 'group{group_size}'
    conv_mode: 'same', 'valid'
    dim: 2, 3
    up_mode: 'transposed', 'nearest', 'linear', 'bilinear', 'bicubic', 'trilinear'
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 2,
        n_blocks: int = 4,
        start_filters: int = 32,
        activation: str = ActivationFunction.RELU,
        normalization: str = NormalizationLayer.BATCH,
        conv_mode: str = ConvMode.SAME,
        dim: int = Dimensions.TWO,
        up_mode: str = UpMode.TRANSPOSED,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_blocks = n_blocks
        self.start_filters = start_filters
        self.activation = activation
        self.normalization = normalization
        self.conv_mode = conv_mode
        self.dim = dim
        self.up_mode = up_mode

        self.down_blocks = []
        self.up_blocks = []

        # create encoder path
        for i in range(self.n_blocks):
            num_filters_in = self.in_channels if i == 0 else num_filters_out
            num_filters_out = self.start_filters * (2**i)
            pooling = True if i < self.n_blocks - 1 else False

            down_block = DownBlock(
                in_channels=num_filters_in,
                out_channels=num_filters_out,
                pooling=pooling,
                activation=self.activation,
                normalization=self.normalization,
                conv_mode=self.conv_mode,
                dim=self.dim,
            )

            self.down_blocks.append(down_block)

        # create decoder path (requires only n_blocks-1 blocks)
        for i in range(n_blocks - 1):
            num_filters_in = num_filters_out
            num_filters_out = num_filters_in // 2

            up_block = UpBlock(
                in_channels=num_filters_in,
                out_channels=num_filters_out,
                activation=self.activation,
                normalization=self.normalization,
                conv_mode=self.conv_mode,
                dim=self.dim,
                up_mode=self.up_mode,
            )

            self.up_blocks.append(up_block)

        # final convolution
        self.conv_final = get_conv_layer(
            num_filters_out,
            self.out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
            dim=self.dim,
        )

        # add the list of modules to current module
        self.down_blocks = nn.ModuleList(self.down_blocks)
        self.up_blocks = nn.ModuleList(self.up_blocks)

        # initialize the weights
        self.initialize_parameters()

    @staticmethod
    def weight_init(module, method, **kwargs):
        if isinstance(
            module, (nn.Conv3d, nn.Conv2d, nn.ConvTranspose3d, nn.ConvTranspose2d)
        ):
            method(module.weight, **kwargs)  # weights

    @staticmethod
    def bias_init(module, method, **kwargs):
        if isinstance(
            module, (nn.Conv3d, nn.Conv2d, nn.ConvTranspose3d, nn.ConvTranspose2d)
        ):
            method(module.bias, **kwargs)  # bias

    def initialize_parameters(
        self, method_weights=nn.init.xavier_uniform_, method_bias=nn.init.zeros_
    ):
        for module in self.modules():
            self.weight_init(module, method_weights)  # initialize weights
            self.bias_init(module, method_bias)  # initialize bias

    def forward(self, x: torch.tensor):
        encoder_output = []

        # Encoder pathway
        for module in self.down_blocks:
            x, before_pooling = module(x)
            encoder_output.append(before_pooling)

        # Decoder pathway
        for i, module in enumerate(self.up_blocks):
            before_pool = encoder_output[-(i + 2)]
            x = module(before_pool, x)

        x = self.conv_final(x)

        return x

    def __repr__(self):
        attributes = {
            attr_key: self.__dict__[attr_key]
            for attr_key in self.__dict__.keys()
            if "_" not in attr_key[0] and "training" not in attr_key
        }
        d = {self.__class__.__name__: attributes}
        return f"{d}"


if __name__ == "__main__":
    unet = UNet(
        in_channels=1,
        out_channels=2,
        n_blocks=4,
        start_filters=32,
        activation=ActivationFunction.RELU,
        normalization=NormalizationLayer.BATCH,
        conv_mode=ConvMode.SAME,
        dim=Dimensions.TWO,
        up_mode=UpMode.TRANSPOSED,
    )
    from torchinfo import summary

    # [B, C, H, W]
    summary = summary(model=unet, input_size=(1, 1, 512, 512), device="cpu")
