from __future__ import division, print_function

from bokeh.plotting import figure
from bokeh.models import Range1d


def plot_image_batch(
    batch,
    fig=None,
    nrows=10,
    ncols=10,
):
    """ Plots the first 100 samples of a batch of images in a 10x10 grid.

    # TODO set figure size based on nrows, ncols

    Parameters
    ----------
    batch : np.array(shape=(<N>,0,28,28)), where <N> must be larger than 100
    figure_ : bokeh.plotting.Figure | None

    Returns
    -------
    fig : bokeh.plotting.Figure
    """
    if fig is None:
        fig = figure()
    fig.image(
        [im for im in batch[:nrows*ncols, 0, ::-1, :]],
        x=range(ncols) * nrows,
        y=sum([[i]*ncols for i in range(nrows)], []),
        dw=[1] * 100,
        dh=[1] * 100
    )
    fig.x_range = Range1d(0, ncols)
    fig.y_range = Range1d(0, nrows)

    return fig