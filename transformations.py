import numpy as np
from skimage.transform import resize
from sklearn.externals._pilutil import bytescale


def create_dense_target(tar: np.ndarray):
    classes = np.unique(tar)
    dummy = np.zeros_like(tar)
    for idx, value in enumerate(classes):
        mask = np.where(tar == value)
        dummy[mask] = idx

    return dummy


def normalize_01(inp: np.ndarray):
    inp_out = (inp - np.min(inp)) / np.ptp(inp)
    return inp_out


def normalize(inp: np.ndarray, mean: float, std: float):
    inp_out = (inp - mean) / std
    return inp_out


def re_normalize(inp: np.ndarray,
                 low: int = 0,
                 high: int = 255
                 ):
    """Normalize the data to a certain range. Default: [0-255]"""
    inp_out = bytescale(inp, low=low, high=high)
    return inp_out


class Compose:
    """
    Composes several transforms together.
    """

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, inp, target):
        for t in self.transforms:
            inp, target = t(inp, target)
        return inp, target

    def __repr__(self): return str([transform for transform in self.transforms])


class MoveAxis:
    """From [H, W, C] to [C, H, W]"""

    def __init__(self, transform_input: bool = True, transform_target: bool = False):
        self.transform_input = transform_input
        self.transform_target = transform_target

    def __call__(self, inp: np.ndarray, tar: np.ndarray):
        if self.transform_input: inp = np.moveaxis(inp, -1, 0)
        if self.transform_target: tar = np.moveaxis(inp, -1, 0)

        return inp, tar

    def __repr__(self):
        return str({self.__class__.__name__: self.__dict__})


class DenseTarget:
    """Creates segmentation maps with consecutive integers, starting from 0"""

    def __init__(self):
        pass

    def __call__(self, inp: np.ndarray, tar: np.ndarray):
        tar = create_dense_target(tar)

        return inp, tar

    def __repr__(self):
        return str({self.__class__.__name__: self.__dict__})


class Resize:
    """Resizes the image and target - based on skimage"""

    def __init__(self,
                 input_size: tuple,
                 target_size: tuple,
                 input_kwargs: dict = {},
                 target_kwargs: dict = {'order': 0, 'anti_aliasing': False, 'preserve_range': True}
                 ):
        self.input_size = input_size
        self.target_size = target_size
        self.input_kwargs = input_kwargs
        self.target_kwargs = target_kwargs

    def __call__(self, inp: np.ndarray, tar: np.ndarray):
        self.input_dtype = inp.dtype
        self.target_dtype = tar.dtype

        inp_out = resize(image=inp,
                         output_shape=self.input_size,
                         **self.input_kwargs
                         )
        tar_out = resize(image=tar,
                         output_shape=self.target_size,
                         **self.target_kwargs
                         ).astype(self.target_dtype)
        return inp_out, tar_out

    def __repr__(self):
        return str({self.__class__.__name__: self.__dict__})


class Normalize01:
    """Squash image input to the value range [0, 1] (no clipping)"""

    def __init__(self):
        pass

    def __call__(self, inp, tar):
        inp = normalize_01(inp)

        return inp, tar

    def __repr__(self):
        return str({self.__class__.__name__: self.__dict__})


class Normalize:
    """Normalize based on mean and standard deviation."""

    def __init__(self,
                 mean: float,
                 std: float,
                 transform_input=True,
                 transform_target=False
                 ):
        self.transform_input = transform_input
        self.transform_target = transform_target
        self.mean = mean
        self.std = std

    def __call__(self, inp, tar):
        inp = normalize(inp)

        return inp, tar

    def __repr__(self):
        return str({self.__class__.__name__: self.__dict__})


class AlbuSeg2d:
    def __init__(self, albu):
        self.albu = albu

    def __call__(self, inp, tar):
        # input, target
        out_dict = self.albu(image=inp, mask=tar)
        input_out = out_dict['image']
        target_out = out_dict['mask']

        return input_out, target_out

    def __repr__(self):
        return str({self.__class__.__name__: self.__dict__})