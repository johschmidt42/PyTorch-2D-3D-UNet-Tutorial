import pytest
import torch

from unet import (
    ActivationFunction,
    ConvMode,
    Dimensions,
    NormalizationLayer,
    UNet,
    UpMode,
)


def test_unet_2d():
    batch_size = 1
    in_channels = 1
    out_channels = 2
    height = 256
    width = 256
    unet = UNet(
        in_channels=in_channels,
        out_channels=out_channels,
        n_blocks=4,
        start_filters=32,
        activation=ActivationFunction.RELU,
        normalization=NormalizationLayer.BATCH,
        conv_mode=ConvMode.SAME,
        dim=Dimensions.TWO,
        up_mode=UpMode.TRANSPOSED,
    )

    inp = torch.rand(size=(batch_size, in_channels, height, width), dtype=torch.float32)
    out = unet(inp)
    assert out.shape == (batch_size, out_channels, height, width)


def test_unet_3d():
    batch_size = 1
    in_channels = 1
    out_channels = 2
    height = 64
    width = 64
    depth = 64
    unet = UNet(
        in_channels=in_channels,
        out_channels=out_channels,
        n_blocks=4,
        start_filters=32,
        activation=ActivationFunction.RELU,
        normalization=NormalizationLayer.BATCH,
        conv_mode=ConvMode.SAME,
        dim=Dimensions.THREE,
        up_mode=UpMode.TRANSPOSED,
    )

    inp = torch.rand(
        size=(batch_size, in_channels, depth, height, width), dtype=torch.float32
    )
    out = unet(inp)
    assert out.shape == (batch_size, out_channels, depth, height, width)


def test_unet_valid():
    """
    Same settings & input as in U-Net: Convolutional Networks for Biomedical Image Segmentation: https://arxiv.org/abs/1505.04597
    """
    batch_size = 1
    in_channels = 1
    out_channels = 2
    input_spatial_dim = 572
    expected_spatial_dim = 388
    unet = UNet(
        in_channels=in_channels,
        out_channels=out_channels,
        n_blocks=5,
        start_filters=32,
        activation=ActivationFunction.RELU,
        normalization=NormalizationLayer.BATCH,
        conv_mode=ConvMode.VALID,
        dim=Dimensions.TWO,
        up_mode=UpMode.TRANSPOSED,
    )

    inp = torch.rand(
        size=(batch_size, in_channels, input_spatial_dim, input_spatial_dim),
        dtype=torch.float32,
    )
    out = unet(inp)
    assert out.shape == (
        batch_size,
        out_channels,
        expected_spatial_dim,
        expected_spatial_dim,
    )


@pytest.mark.parametrize(
    argnames="up_mode", argvalues=[UpMode.BILINEAR, UpMode.BICUBIC]
)
def test_unet_2d_up_modes(up_mode):
    batch_size = 1
    in_channels = 1
    out_channels = 2
    height = 256
    width = 256
    unet = UNet(
        in_channels=in_channels,
        out_channels=out_channels,
        n_blocks=4,
        start_filters=32,
        activation=ActivationFunction.RELU,
        normalization=NormalizationLayer.BATCH,
        conv_mode=ConvMode.SAME,
        dim=Dimensions.TWO,
        up_mode=up_mode,
    )

    inp = torch.rand(size=(batch_size, in_channels, height, width), dtype=torch.float32)
    out = unet(inp)
    assert out.shape == (batch_size, out_channels, height, width)


@pytest.mark.parametrize(argnames="up_mode", argvalues=[UpMode.TRILINEAR])
def test_unet_3d_up_modes(up_mode):
    batch_size = 1
    in_channels = 1
    out_channels = 2
    height = 64
    width = 64
    depth = 64
    unet = UNet(
        in_channels=in_channels,
        out_channels=out_channels,
        n_blocks=4,
        start_filters=32,
        activation=ActivationFunction.RELU,
        normalization=NormalizationLayer.BATCH,
        conv_mode=ConvMode.SAME,
        dim=Dimensions.THREE,
        up_mode=up_mode,
    )

    inp = torch.rand(
        size=(batch_size, in_channels, depth, height, width), dtype=torch.float32
    )
    out = unet(inp)
    assert out.shape == (batch_size, out_channels, depth, height, width)
