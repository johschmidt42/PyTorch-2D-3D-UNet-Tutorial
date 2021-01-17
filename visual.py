import numpy as np


def show_input_target_pair_napari(gen_training, gen_validation=None):
    """
    Press 't' to get a random sample of the next training batch.
    Press 'v' to get a random sample of the next validation batch.
    """
    # Batch
    x, y = next(iter(gen_training))

    # Napari
    import napari
    with napari.gui_qt():
        viewer = napari.Viewer()
        img = viewer.add_image(x, name='input_training')
        tar = viewer.add_labels(y, name='target_training')

        @viewer.bind_key('t')
        def next_batch_training(viewer):
            x, y = next(iter(gen_training))
            img.data = x
            tar.data = y
            img.name = 'input_training'
            tar.name = 'target_training'

        if gen_validation:
            @viewer.bind_key('v')
            def next_batch_validation(viewer):
                x, y = next(iter(gen_validation))
                img.data = x
                tar.data = y
                img.name = 'input_validation'
                tar.name = 'target_validation'

    return viewer


class Input_Target_Pair_Generator:
    def __init__(self,
                 dataloader,
                 re_normalize=True,
                 rgb=False,
                 ):
        self.dataloader = dataloader
        self.re_normalize = re_normalize
        self.rgb = rgb

    def __iter__(self):
        return self

    def __next__(self):
        x, y = next(iter(self.dataloader))
        x, y = x.cpu().numpy(), y.cpu().numpy()  # make sure it's a numpy.ndarray on the cpu

        # Batch
        batch_size = x.shape[0]
        rand_num = np.random.randint(low=0, high=batch_size)
        x, y = x[rand_num], y[rand_num]  # Pick a random image from the batch

        # RGB
        if self.rgb:
            x = np.moveaxis(x, source=0, destination=-1)  # from [C, H, W] to [H, W, C]

        # Re-normalize
        if self.re_normalize:
            from transformations import re_normalize
            x = re_normalize(x)

        return x, y


def plot_training(training_losses,
                  validation_losses,
                  learning_rate,
                  gaussian=True,
                  sigma=2,
                  figsize=(8, 6)
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
        subfig.spines['top'].set_visible(False)
        subfig.spines['right'].set_visible(False)

    if gaussian:
        training_losses_gauss = gaussian_filter(training_losses, sigma=sigma)
        validation_losses_gauss = gaussian_filter(validation_losses, sigma=sigma)

        linestyle_original = '.'
        color_original_train = 'lightcoral'
        color_original_valid = 'lightgreen'
        color_smooth_train = 'red'
        color_smooth_valid = 'green'
        alpha = 0.25
    else:
        linestyle_original = '-'
        color_original_train = 'red'
        color_original_valid = 'green'
        alpha = 1.0

    # Subfig 1
    subfig1.plot(x_range, training_losses, linestyle_original, color=color_original_train, label='Training',
                 alpha=alpha)
    subfig1.plot(x_range, validation_losses, linestyle_original, color=color_original_valid, label='Validation',
                 alpha=alpha)
    if gaussian:
        subfig1.plot(x_range, training_losses_gauss, '-', color=color_smooth_train, label='Training', alpha=0.75)
        subfig1.plot(x_range, validation_losses_gauss, '-', color=color_smooth_valid, label='Validation', alpha=0.75)
    subfig1.title.set_text('Training & validation loss')
    subfig1.set_xlabel('Epoch')
    subfig1.set_ylabel('Loss')

    subfig1.legend(loc='upper right')

    # Subfig 2
    subfig2.plot(x_range, learning_rate, color='black')
    subfig2.title.set_text('Learning rate')
    subfig2.set_xlabel('Epoch')
    subfig2.set_ylabel('LR')

    return fig
