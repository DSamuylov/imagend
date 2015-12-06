"""A small python package to visualize 3 and 4 dimensional image data.

Author: Denis Samuylov
E-mail: denis.samuylov@gmail.com

"""

import math
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as font_manager


class DimensionError(Exception):

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr("The image has a wrong dimension ({})".format(self.value))


def project(img, axis, proj='mean'):
    """Mean|Median|Max projection of an image along a given axis"""
    assert proj in ['mean', 'median', 'max']
    if proj is 'mean':
        return np.mean(img, axis=axis)
    elif proj is 'median':
        return np.median(img, axis=axis)
    elif proj is 'max':
        return np.max(img, axis=axis)


def project_image_stack(img, vmin=None, vmax=None, axes=None,
                  proj='mean', figwidth=3, add_labels=False,
                  cmap='gray'):
    """Plot x-, y-, z-projections of an image stack.

    Parameters
    ----------
    img : array
        3D stack of images
    vmin, vmax : scalar, optional, default: None
        vmin and vmax are used to normalize luminance data. By default it takes
        all range of gray values of computed projections.
    axes : list, optional, default : None
        It is possible to pass axes used to plot projection.
    proj : str, optional, default : 'mean'
        It defines the type of projection that we would like to plot. It can
        be: [mean|median|max]
    figwidth : scalar
        The figure width is in inches. The height is computed automatically to
        preserve the aspect ratio.
    add_labels : boolean
        If true, write a label for each projection.

    Returns
    -------
    fig : figure
        Created figure with 3 projections. It is None when axes are provided as
        an input.
    axes : tuple
        A tuple of axes for z, y, x projections respectively.

    """
    img = np.array(img)
    dim = len(img.shape)
    if dim != 3:
        raise DimensionError(dim)
    n_slices, n_h, n_w = img.shape

    figheight = (figwidth)*np.float64(n_h + n_slices)/(n_w + n_slices)

    z_proj = project(img, axis=0, proj=proj)
    y_proj = project(img, axis=1, proj=proj)
    x_proj = project(img, axis=2, proj=proj).T

    if vmin is None:
        vmin = min([z_proj.min(), y_proj.min(), x_proj.min()])
    if vmax is None:
        vmax = max([z_proj.max(), y_proj.max(), x_proj.max()])

    if axes is None:
        # Create figure
        fig = plt.figure(figsize=(figwidth, figheight))
        gs = gridspec.GridSpec(2, 2, width_ratios=[n_w, n_slices],
                               height_ratios=[n_h, n_slices])
        gs.update(wspace=0.05, hspace=0.05, bottom=0, top=1, left=0, right=1)
        # Create axes:
        ax_z = fig.add_subplot(gs[0, 0])
        ax_y = fig.add_subplot(gs[1, 0], sharex=ax_z)
        ax_x = fig.add_subplot(gs[0, 1], sharey=ax_z)
    else:
        assert len(axes) == 3
        fig = None
        ax_z, ax_y, ax_x = axes
    # z projection:
    ax_z.imshow(z_proj, interpolation='nearest', cmap=cmap, aspect=1,
                vmin=vmin, vmax=vmax)
    ax_z.set_xlim([-0.5, n_w - 0.5])
    ax_z.set_ylim([n_h - 0.5, -0.5])
    ax_z.axis('off')
    # y projection:
    ax_y.imshow(y_proj, interpolation='nearest', cmap=cmap, aspect=1,
                vmin=vmin, vmax=vmax)
    ax_y.set_xlim([-0.5, n_w-0.5])
    ax_y.set_ylim([n_slices-0.5, -0.5])
    ax_y.axis('off')
    # x projection:
    ax_x.imshow(x_proj, interpolation='nearest', cmap=cmap, aspect=1,
                vmin=vmin, vmax=vmax)
    ax_x.set_xlim([-0.5, n_slices-0.5])
    ax_x.set_ylim([n_h-0.5, -0.5])
    ax_x.axis('off')

    if add_labels:
        # Draw xyz labels:
        font = font_manager.FontProperties()
        # font.set_weight('bold')
        font.set_size(15)
        ax_z.text(2, 4, 'z', color='white', fontproperties=font)
        ax_y.text(2, 4, 'y', color='white', fontproperties=font)
        ax_x.text(2, 4, 'x', color='white', fontproperties=font)
    return fig, (ax_z, ax_y, ax_x)


def project_image_sequence(img_sequence, frames=None,
                   nsubfig=None, subfigwidth=3, ncol=5,
                   vmin=None, vmax=None, proj='mean',
                   add_labels=False, cmap=cm.gray):
    """Plot a grid of x,y,z projections of an image sequence.

    Parameters
    ----------
    img_sequence : array
        A 4D array with an image sequence.
    frames : list, optional, default : None
        The list of frame indexes that we would like to plot.
    nsubfig : scalar, optional, default : None
        The number of images to plot. By default: all images will be plotted.
    subfigwidth : scalar
        The figure width is in inches. The height is computed automatically to
        preserve the aspect ratio.
    ncol : scalar
        The number of projections in each row.
    vmin, vmax : scalar, optional, default: None
        vmin and vmax are used to normalize luminance data. By default it takes
        all range of gray values of computed projections.
    projection : str, optional, default : 'mean'
        It defines the type of projection that we would like to plot. It can
        be: [mean|median|max]
    add_labels : boolean
        If true, write a label for each projection.

    Returns
    -------
    fig : figure
        A created figure.
    axes : list
        A list of tuples with axes for each projection.

    """
    img_sequence = np.array(img_sequence)
    dim = len(img_sequence.shape)
    if dim != 4:
        raise DimensionError(dim)
    n_frames, n_slices, n_h, n_w = img_sequence.shape

    if frames is None:
        frames = range(0, n_frames)
    if nsubfig is None:
        n_subfig = len(frames)

    # Compute correct normalization for projected images:
    if vmin is None:
        vmin = project(img_sequence, axis=1, proj=proj).min()
    if vmax is None:
        vmax = project(img_sequence, axis=1, proj=proj).max()

    # Initialize parameters
    n_row = int(math.ceil(float(n_subfig)/ncol))
    height_subfig = (n_h + n_slices)*np.float64(subfigwidth)/(n_w + n_slices)

    # Initialize the figure
    fig = plt.figure(figsize=(subfigwidth*ncol, height_subfig*n_row))

    # Initialize the layout
    gs_master = gridspec.GridSpec(n_row, ncol)
    gs_master.update(bottom=0, top=1, left=0, right=1)

    # Magic for every (sub-)gridspec:
    wspace = 0.05  # [in]
    hspace = wspace*float(n_w + n_slices)/(n_h + n_slices)

    axes = []
    for i in range(n_row):
        for j in range(ncol):
            ind = i*ncol + j
            if ind >= n_subfig:
                break

            gs = gridspec.GridSpecFromSubplotSpec(
                      2, 2,
                      width_ratios=[n_w, n_slices],
                      height_ratios=[n_h, n_slices],
                      subplot_spec=gs_master[i, j],
                      wspace=wspace, hspace=hspace,
                      )

            ax_z = plt.Subplot(fig, gs[0, 0])
            ax_z.set_title('frame={}'.format(frames[ind]))
            ax_y = plt.Subplot(fig, gs[1, 0], sharex=ax_z)
            ax_x = plt.Subplot(fig, gs[0, 1], sharey=ax_z)

            project_image_stack(img_sequence[frames[ind]],
                          axes=[ax_z, ax_y, ax_x],
                          vmin=vmin, vmax=vmax, proj=proj,
                          add_labels=add_labels, cmap=cmap)
            fig.add_subplot(ax_z)
            fig.add_subplot(ax_y)
            fig.add_subplot(ax_x)
            axes.append((ax_z, ax_y, ax_x))
    return fig, axes
