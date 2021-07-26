import napari
import numpy as np

from transformations import re_normalize


def enable_gui_qt():
    """Performs the magic command %gui qt"""
    from IPython import get_ipython

    ipython = get_ipython()
    ipython.magic("gui qt")


class DatasetViewer:
    def __init__(self, dataset):

        self.dataset = dataset
        self.index = 0

        # napari viewer instance
        self.viewer = None

        # current image & shape layer
        self.image_layer = None
        self.label_layer = None

    def napari(self):
        # IPython magic for napari < 0.4.8
        enable_gui_qt()

        # napari
        if self.viewer:
            try:
                del self.viewer
            except AttributeError:
                pass
        self.index = 0

        # Init napari instance
        self.viewer = napari.Viewer()

        # Show current sample
        self.show_sample()

        # Key-bindings
        # Press 'n' to get the next sample
        @self.viewer.bind_key("n")
        def next(viewer):
            self.increase_index()  # Increase the index
            self.show_sample()  # Show next sample

        # Press 'b' to get the previous sample
        @self.viewer.bind_key("b")
        def prev(viewer):
            self.decrease_index()  # Decrease the index
            self.show_sample()  # Show next sample

    def increase_index(self):
        self.index += 1
        if self.index >= len(self.dataset):
            self.index = 0

    def decrease_index(self):
        self.index -= 1
        if self.index < 0:
            self.index = len(self.dataset) - 1

    def show_sample(self):

        # Get a sample from the dataset
        sample = self.get_sample_dataset(self.index)
        x, y = sample

        # Get the names from the dataset
        names = self.get_names_dataset(self.index)
        x_name, y_name = names
        x_name, y_name = x_name.name, y_name.name  # only possible if pathlib.Path

        # Transform the sample to numpy, cpu and correct format to visualize
        x = self.transform_x(x)
        y = self.transform_y(y)

        # Create or update image layer
        if self.image_layer not in self.viewer.layers:
            self.image_layer = self.create_image_layer(x, x_name)
        else:
            self.update_image_layer(self.image_layer, x, x_name)

        # Create or update label layer
        if self.label_layer not in self.viewer.layers:
            self.label_layer = self.create_label_layer(y, y_name)
        else:
            self.update_label_layer(self.label_layer, y, y_name)

        # Reset view
        self.viewer.reset_view()

    def create_image_layer(self, x, x_name):
        return self.viewer.add_image(x, name=str(x_name))

    def update_image_layer(self, image_layer, x, x_name):
        """Replace the data and the name of a given image_layer"""
        image_layer.data = x
        image_layer.name = str(x_name)

    def create_label_layer(self, y, y_name):
        return self.viewer.add_labels(y, name=str(y_name))

    def update_label_layer(self, target_layer, y, y_name):
        """Replace the data and the name of a given image_layer"""
        target_layer.data = y
        target_layer.name = str(y_name)

    def get_sample_dataset(self, index):
        return self.dataset[index]

    def get_names_dataset(self, index):
        return self.dataset.inputs[index], self.dataset.targets[index]

    def transform_x(self, x):
        # make sure it's a numpy.ndarray on the cpu
        x = x.cpu().numpy()

        # from [C, H, W] to [H, W, C] - only for RGB images.
        if self.check_if_rgb(x):
            x = np.moveaxis(x, source=0, destination=-1)

        # Re-normalize
        x = re_normalize(x)

        return x

    def transform_y(self, y):
        # make sure it's a numpy.ndarray on the cpu
        y = y.cpu().numpy()

        return y

    def check_if_rgb(self, x):
        # checks if the shape of the first dim (channel dim) is 3
        # TODO: Try other methods as a 3D grayscale input image can have 3 modalities -> 3 channels
        # TODO: Also think about RGBA images with 4 channels or a combination of a RGB and a grayscale image -> 4 channels
        return True if x.shape[0] == 3 else False


def plot_training(
    training_losses,
    validation_losses,
    learning_rate,
    gaussian=True,
    sigma=2,
    figsize=(8, 6),
):
    """
    Returns a loss plot with training loss, validation loss and learning rate.
    """

    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    from scipy.ndimage import gaussian_filter

    list_len = len(training_losses)
    x_range = list(range(1, list_len + 1))  # number of x values

    fig = plt.figure(figsize=figsize)
    grid = gridspec.GridSpec(ncols=2, nrows=1, figure=fig)

    subfig1 = fig.add_subplot(grid[0, 0])
    subfig2 = fig.add_subplot(grid[0, 1])

    subfigures = fig.get_axes()

    for i, subfig in enumerate(subfigures, start=1):
        subfig.spines["top"].set_visible(False)
        subfig.spines["right"].set_visible(False)

    if gaussian:
        training_losses_gauss = gaussian_filter(training_losses, sigma=sigma)
        validation_losses_gauss = gaussian_filter(validation_losses, sigma=sigma)

        linestyle_original = "."
        color_original_train = "lightcoral"
        color_original_valid = "lightgreen"
        color_smooth_train = "red"
        color_smooth_valid = "green"
        alpha = 0.25
    else:
        linestyle_original = "-"
        color_original_train = "red"
        color_original_valid = "green"
        alpha = 1.0

    # Subfig 1
    subfig1.plot(
        x_range,
        training_losses,
        linestyle_original,
        color=color_original_train,
        label="Training",
        alpha=alpha,
    )
    subfig1.plot(
        x_range,
        validation_losses,
        linestyle_original,
        color=color_original_valid,
        label="Validation",
        alpha=alpha,
    )
    if gaussian:
        subfig1.plot(
            x_range,
            training_losses_gauss,
            "-",
            color=color_smooth_train,
            label="Training",
            alpha=0.75,
        )
        subfig1.plot(
            x_range,
            validation_losses_gauss,
            "-",
            color=color_smooth_valid,
            label="Validation",
            alpha=0.75,
        )
    subfig1.title.set_text("Training & validation loss")
    subfig1.set_xlabel("Epoch")
    subfig1.set_ylabel("Loss")

    subfig1.legend(loc="upper right")

    # Subfig 2
    subfig2.plot(x_range, learning_rate, color="black")
    subfig2.title.set_text("Learning rate")
    subfig2.set_xlabel("Epoch")
    subfig2.set_ylabel("LR")

    return fig
