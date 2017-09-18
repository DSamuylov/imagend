"""Python package to visualise image volumes.

Author: Denis Samuylov, Prateek Purwar
E-mail: denis.samuylov@gmail.com, pp.semtex@gmail.com

"""

import math
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatch


class DimensionError(Exception):

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr("Wrong dimension! ({})".format(self.value))


def project_along_axis(image, axis, method="mean"):
    """Project image stack along an axis.

    Parameters
    ----------
    image: np.array
        Image data.
    axis: int
        Dimension.
    method: str
        Define the projection method.

    """
    assert method in ["mean", "median", "max"]
    if method == "mean":
        return np.mean(image, axis=axis)
    elif method == "median":
        return np.median(image, axis=axis)
    elif method == "max":
        return np.max(image, axis=axis)
    else:
        raise Exception("Unknown projection method!")


def compute_projected_normalisation(image, method="mean"):
    """Compute the normalization of a projected image stack.

    Parameters
    ----------
    image: np.array
        3D or 4D image data.
    method: str
        Define the projection method.

    """
    dim = len(image.shape)
    if dim == 3:
        image = image.reshape((1,) + image.shape)
    if dim != 4:
        raise DimensionError()

    proj_list = [project_along_axis(image_stack, ax_id, method)
                 for image_stack in image for ax_id in range(3)]
    vmin = min([_proj.min() for _proj in proj_list])
    vmax = max([_proj.max() for _proj in proj_list])
    return vmin, vmax


def project_image_stack(image_stack, vmin=None, vmax=None, method="mean",
                        figwidth=3, cmap=cm.magma_r, alpha=1, aspect=1,
                        projections=["z", "y", "x"], axes=None):
    """Plot orthogonal projections of an image stack.

    Parameters
    ----------
    image_stack : array
        Three dimensional image volume.
    vmin, vmax : scalar
        Normalization of the luminance data. By default it takes all range of
        gray values of computed projections.
    method : str
        It defines the type of projection that we would like to plot. It can
        be: [mean|median|max]
    figwidth : scalar
        The figure width is in inches. The height is computed automatically to
        preserve the aspect ratio.
    add_labels : boolean
        If true, write a label for each projection.
    projections: list
        List of orthogonal projections to plot.
    axes : list
        Axes used to plot projection.

    Returns
    -------
    fig : figure
        Created figure with 3 projections. It is None when axes are provided as
        an input.
    axes : tuple
        A tuple of axes for z, y, x projections respectively.

    """
    image_stack = np.array(image_stack)
    dim = len(image_stack.shape)
    if dim != 3:
        raise DimensionError(dim)
    n_s, n_h, n_w = image_stack.shape

    z_proj = project_along_axis(image_stack, axis=0, method=method)
    y_proj = project_along_axis(image_stack, axis=1, method=method)
    x_proj = project_along_axis(image_stack, axis=2, method=method).T

    if vmin is None:
        vmin = min([z_proj.min(), y_proj.min(), x_proj.min()])
    if vmax is None:
        vmax = max([z_proj.max(), y_proj.max(), x_proj.max()])

    ax_z, ax_y, ax_x = [None, None, None]
    if axes is None:
        if ("z" in projections) & ("y" in projections) & ("x" in projections):
            figheight = (figwidth)*np.float64(n_h + n_s)/(n_w + n_s)
            # Create figure:
            fig = plt.figure(figsize=(figwidth, figheight))
            # Create grid:
            gs = gridspec.GridSpec(2, 2, width_ratios=[n_w, n_s],
                                   height_ratios=[n_h, n_s])
            gs.update(wspace=0.05, hspace=0.05,
                      bottom=0, top=1, left=0, right=1)
            # Create axes:
            ax_z = fig.add_subplot(gs[0, 0])
            ax_y = fig.add_subplot(gs[1, 0], sharex=ax_z)
            ax_x = fig.add_subplot(gs[0, 1], sharey=ax_z)
        if (("z" in projections) &
                ("x" in projections) &
                ("y" not in projections)):
            figheight = (figwidth)*np.float64(n_h)/(n_w + n_s)
            # Create figure
            fig = plt.figure(figsize=(figwidth, figheight))
            # Create grid:
            gs = gridspec.GridSpec(nrows=1, ncols=2,
                                   width_ratios=[n_w, n_s],
                                   height_ratios=[n_h])
            gs.update(wspace=0.05, hspace=0.05,
                      bottom=0, top=1, left=0, right=1)
            # Create axes:
            ax_z = fig.add_subplot(gs[0, 0])
            ax_x = fig.add_subplot(gs[0, 1], sharey=ax_z)
    else:
        assert (len(axes) == 3) or (len(axes) == 2)
        fig = None
        for i, method in enumerate(projections):
            if method == "z":
                ax_z = axes[i]
            elif method == "y":
                ax_y = axes[i]
            elif method == "x":
                ax_x = axes[i]
            else:
                raise Exception("Axis label is wrong!")

    if "z" in projections:
        ax_z.imshow(z_proj, interpolation="nearest", cmap=cmap, aspect=1,
                    vmin=vmin, vmax=vmax, alpha=alpha)
        ax_z.set_xlim([-0.5, n_w - 0.5])
        ax_z.set_ylim([n_h - 0.5, -0.5])
        ax_z.axis("off")
    if "y" in projections:
        ax_y.imshow(y_proj, interpolation="nearest", cmap=cmap, aspect=1,
                    vmin=vmin, vmax=vmax, alpha=alpha)
        ax_y.set_xlim([-0.5, n_w-0.5])
        ax_y.set_ylim([n_s-0.5, -0.5])
        ax_y.axis("off")
        ax_y.set_aspect(aspect)
    if "x" in projections:
        ax_x.imshow(x_proj, interpolation="nearest", cmap=cmap, aspect=1,
                    vmin=vmin, vmax=vmax, alpha=alpha)
        ax_x.set_xlim([-0.5, n_s-0.5])
        ax_x.set_ylim([n_h-0.5, -0.5])
        ax_x.axis("off")
        ax_x.set_aspect(1/aspect)

    axes = []
    for method in projections:
        if method == "z":
            axes.append(ax_z)
        if method == "y":
            axes.append(ax_y)
        if method == "x":
            axes.append(ax_x)
    return fig, axes


def show_image_stack(
        image_stack, slices=None, labels=None, n_cols=5, subfigwidth=2,
        vmin=None, vmax=None, normalized=True, cmap=cm.magma_r):
    """Display slices of a three dimensional image stack."""
    dim = len(image_stack.shape)
    if dim != 3:
        raise DimensionError(dim)
    n_slices, n_height, n_width = image_stack.shape

    if slices is None:
        slices = np.arange(n_slices)
    if type(slices) == int:
        slices = np.linspace(0, n_slices - 1, slices, dtype=np.uint8)

    if labels is None:
        labels = [None]*n_slices

    # Number of subfigures:
    n_images = len(slices)
    n_rows = int(np.ceil(float(n_images)/n_cols))

    # Gray value range:
    vmin = image_stack.min() if normalized & (vmin is None) else vmin
    vmax = image_stack.max() if normalized & (vmax is None) else vmax

    fig, axes = plt.subplots(
        nrows=n_rows, ncols=n_cols,
        figsize=(subfigwidth*n_cols, subfigwidth*n_rows))
    for index in range(n_cols*n_rows):
        i_col = index % n_cols
        i_row = (index - i_col + 1)/n_cols

        if (n_rows > 1) & (n_cols > 1):
            ax = axes[i_row, i_col]
        elif (n_cols > 1):
            ax = axes[i_col]
        else:
            ax = axes

        ax.axis("off")

        if index < n_images:
            slice_id = slices[index]
            image = image_stack[slice_id]
            label = labels[index]
            ax.imshow(
                image, interpolation="nearest", cmap=cmap,
                vmin=vmin, vmax=vmax)
            if label is not None:
                ax.set_title(label)

    if n_images == 1:
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    return fig, axes


def project_image_sequence(image_sequence, frames=None, labels=None,
                           method="mean", n_col=5, subfigwidth=3,
                           vmin=None, vmax=None, normalise=True,
                           cmap=cm.magma_r):
    """Plot a grid of x,y,z projections of an image sequence.

    Parameters
    ----------
    image_sequence : array
        A 4D array with an image sequence.
    frames : list, optional, default : None
        The list of frame indexes that we would like to plot.
    subfigwidth : scalar
        The figure width is in inches. The height is computed automatically to
        preserve the aspect ratio.
    n_col : scalar
        The number of projections in each row.
    vmin, vmax : scalar
        Normalization of the luminance data. By default it takes all range of
        gray values of computed projections.
    projection : str, optional, default : "mean"
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
    image_sequence = np.array(image_sequence)
    dim = len(image_sequence.shape)
    if dim != 4:
        raise DimensionError(dim)
    n_frames, n_slices, n_h, n_w = image_sequence.shape

    if frames is None:
        frames = range(0, n_frames)
    elif type(frames) == int:
        frames = np.linspace(0, n_frames - 1, frames,
                             endpoint=True, dtype=np.uint8)
    if labels is None:
        labels = [None]*n_frames
    n_frames_to_show = len(frames)

    # Compute correct normalization for projected images:
    if normalise is True:
        vmin, vmax = compute_projected_normalisation(
            image_sequence, method=method)
    else:
        vmin, vmax = None, None

    # Initialise figure parameters:
    n_row = int(math.ceil(float(len(frames))/n_col))
    subfigheight = (n_h + n_slices)*np.float64(subfigwidth)/(n_w + n_slices)

    # Initialise figure:
    fig = plt.figure(figsize=(subfigwidth*n_col, subfigheight*n_row))
    gs_master = gridspec.GridSpec(n_row, n_col)
    gs_master.update(bottom=0, top=1, left=0, right=1)

    # Magic for every (sub-)gridspec:
    wspace = 0.05  # [in]
    hspace = wspace*float(n_w + n_slices)/(n_h + n_slices)

    axes = []
    for i in range(n_row):
        for j in range(n_col):
            ind = i*n_col + j
            if ind >= n_frames_to_show:
                break
            image_stack = image_sequence[frames[ind]]
            label = labels[ind]

            gs = gridspec.GridSpecFromSubplotSpec(
                2, 2,
                width_ratios=[n_w, n_slices],
                height_ratios=[n_h, n_slices],
                subplot_spec=gs_master[i, j],
                wspace=wspace, hspace=hspace)

            ax_z = plt.Subplot(fig, gs[0, 0])
            if label is not None:
                ax_z.set_title(label)
            ax_y = plt.Subplot(fig, gs[1, 0], sharex=ax_z)
            ax_x = plt.Subplot(fig, gs[0, 1], sharey=ax_z)

            project_image_stack(
                image_stack,
                axes=[ax_z, ax_y, ax_x],
                vmin=vmin, vmax=vmax, method=method, cmap=cmap)
            fig.add_subplot(ax_z)
            fig.add_subplot(ax_y)
            fig.add_subplot(ax_x)
            axes.append((ax_z, ax_y, ax_x))
    return fig, axes


def _project_hdr(mask, axes=None, label=None, figwidth=3, alpha=1, aspect=1,
                 fill=False, projections=["z", "y", "x"], **kwargs):
    assert mask.shape[-1] in [4]
    assert len(mask.shape) == 4
    assert len(projections) > 1
    n_s, n_h, n_w = mask.shape[:-1]

    ax_z, ax_y, ax_x = [None, None, None]
    if axes is None:
        if ("z" in projections) & ("y" in projections) & ("x" in projections):
            figheight = (figwidth)*np.float64(n_h + n_s)/(n_w + n_s)
            # Create figure:
            fig = plt.figure(figsize=(figwidth, figheight))
            # Create grid:
            gs = gridspec.GridSpec(nrows=2, ncols=2,
                                   width_ratios=[n_w, n_s],
                                   height_ratios=[n_h, n_s])
            gs.update(wspace=0.05, hspace=0.05,
                      bottom=0, top=1, left=0, right=1)
            # Create axes:
            ax_z = fig.add_subplot(gs[0, 0])
            ax_y = fig.add_subplot(gs[1, 0], sharex=ax_z)
            ax_x = fig.add_subplot(gs[0, 1], sharey=ax_z)
        if (("z" in projections) &
                ("y" not in projections) & ("x" in projections)):
            figheight = (figwidth)*np.float64(n_h)/(n_w + n_s)
            # Create figure
            fig = plt.figure(figsize=(figwidth, figheight))
            # Create grid:
            gs = gridspec.GridSpec(nrows=1, ncols=2,
                                   width_ratios=[n_w, n_s],
                                   height_ratios=[n_h])
            gs.update(wspace=0.05, hspace=0.05,
                      bottom=0, top=1, left=0, right=1)
            # Create axes:
            ax_z = fig.add_subplot(gs[0, 0])
            ax_x = fig.add_subplot(gs[0, 1], sharey=ax_z)
    else:
        assert (len(axes) == 3) or (len(axes) == 2)
        fig = None
        for i, method in enumerate(projections):
            if method == "z":
                ax_z = axes[i]
            elif method == "y":
                ax_y = axes[i]
            elif method == "x":
                ax_x = axes[i]
            else:
                raise Exception("Axis label is wrong!")

    if fill is False:

        mask_empty = np.zeros((n_s, n_h, n_w, 4))
        if ax_z is not None:
            ax_z.imshow(
                project_along_axis(mask_empty, axis=0))
        if ax_y is not None:
            ax_y.imshow(
                project_along_axis(mask_empty, axis=1))
        if ax_x is not None:
            ax_z.imshow(
                np.swapaxes(project_along_axis(mask_empty, axis=2), 0, 1))

        _y = np.arange(n_h)
        _x = np.arange(n_w)
        _z = np.arange(n_s)

        _mask = mask.max(axis=-1) > 0
        _color_rgba = mask.reshape(-1, 4).max(axis=0)
        _rgb, _alpha = _color_rgba[0:3], _color_rgba[3]

        if ax_z is not None:
            y_list, x_list = np.meshgrid(_y, _x, indexing="ij")
            z_proj = np.zeros((n_h, n_w), np.uint8)
            z_proj[np.max(_mask, axis=0)] = 1
            ax_z.contour(
                x_list, y_list, z_proj,
                levels=[0, 1], colors=[_rgb], alpha=_alpha, origin="image",
                **kwargs)

        if ax_y is not None:
            z_list, x_list = np.meshgrid(_z, _x, indexing="ij")
            y_proj = np.zeros((n_s, n_w), np.uint8)
            y_proj[np.max(_mask, axis=1)] = 1
            ax_y.contour(
                x_list, z_list, y_proj,
                levels=[0, 1], colors=[_rgb], alpha=_alpha, origin="lower",
                **kwargs)

        if ax_x is not None:
            x_list, z_list = np.meshgrid(_y, _z, indexing="ij")
            x_proj = np.zeros((n_h, n_s), np.uint8)
            x_proj[np.swapaxes(np.max(_mask, axis=2), 0, 1)] = 1
            ax_x.contour(
                z_list, x_list, x_proj,
                levels=[0, 1], colors=[_rgb], alpha=_alpha, origin="lower",
                **kwargs)

    if ax_z is not None:
        if label is not None:
            ax_z.set_title(label)
        ax_z.set_xlim([-0.5, n_w - 0.5])
        ax_z.set_ylim([n_h - 0.5, -0.5])
        ax_z.axis("off")
        ax_z.set_aspect(aspect)
    if ax_y is not None:
        ax_y.set_xlim([-0.5, n_w-0.5])
        ax_y.set_ylim([n_s-0.5, -0.5])
        ax_y.axis("off")
        ax_y.set_aspect(aspect)
    if ax_x is not None:
        ax_x.set_xlim([-0.5, n_s-0.5])
        ax_x.set_ylim([n_h-0.5, -0.5])
        ax_x.axis("off")
        ax_x.set_aspect(aspect)

    axes = []
    for method in projections:
        if method == "z":
            axes.append(ax_z)
        if method == "y":
            axes.append(ax_y)
        if method == "x":
            axes.append(ax_x)
    return fig, axes


def project_hdr_stack(image_stack, levels=[0.95, 0.99], label=None, axes=None,
                      colors=None, fill=False, threshold=10e-10, figwidth=3,
                      projections=["z", "y", "x"], **kwargs):
    """Project high density regions.

    Ref: "Computing and Graphing Highest Density Regions" by Rob J. Hyndman

    """
    levels = sorted(levels)
    n_levels = len(levels)
    if colors is None:
        colors = cm.magma_r(np.linspace(0.1, 1, n_levels))

    # print "Fill hdr: {}".format(fill)
    # print "Range of image_stack: {} .. {}".format(
    #     image_stack.min(), image_stack.max())
    # print "Threshold value: {}".format(threshold)

    _img_stack = image_stack.copy()
    indexes = np.where(_img_stack > _img_stack.min() + threshold)
    vals = _img_stack[indexes]
    # print "Number of threshold pixels: {} (out of {})".format(
    #     vals.size, image_stack.size)
    n_px = vals.size

    f_alpha_list = np.unique(vals)
    p_list = np.zeros(f_alpha_list.size)
    for i, f_alpha in enumerate(f_alpha_list):
        p_list[i] = np.sum(image_stack >= f_alpha, dtype=np.float64)/n_px
    alpha_list = 1. - p_list

    hdr_list = np.zeros((n_levels,) + image_stack.shape + (4,))
    for level_id, level in enumerate(levels):
        i = np.squeeze(np.where(alpha_list >= level)).reshape(-1)[0]
        f_alpha = f_alpha_list[i]
        mask = image_stack >= f_alpha
        hdr_list[level_id, mask > 0] = colors[level_id]

    for i, hdr in enumerate(hdr_list):
        _fig, axes = _project_hdr(
            hdr, axes=axes, label=label, fill=fill, figwidth=figwidth,
            projections=projections,
            **kwargs)
        if i == 0:
            fig = _fig
    return fig, axes


def project_hdr_sequence(
        image_sequence, frames=None, levels=[0.95, 0.99], labels=None,
        colors=None, fill=False, threshold=10e-10,
        n_col=5, subfigwidth=3, **kwargs):
    image_sequence = np.array(image_sequence)
    dim = len(image_sequence.shape)
    if dim != 4:
        raise DimensionError(dim)
    n_frames, n_slices, n_h, n_w = image_sequence.shape

    if frames is None:
        frames = range(0, n_frames)
    elif type(frames) == int:
        frames = np.linspace(0, n_frames - 1, frames,
                             endpoint=False, dtype=np.uint8)

    if labels is None:
        labels = [None]*n_frames

    # Initialise figure parameters
    n_row = int(math.ceil(float(n_frames)/n_col))
    subfigheight = (n_h + n_slices)*np.float64(subfigwidth)/(n_w + n_slices)

    # Initialise figure:
    fig = plt.figure(figsize=(subfigwidth*n_col, subfigheight*(n_row)))
    gs_master = gridspec.GridSpec(n_row, n_col)
    gs_master.update(bottom=0, top=1, left=0, right=1)

    # Magic for every (sub-)gridspec:
    wspace = 0.05  # [in]
    hspace = wspace*float(n_w + n_slices)/(n_h + n_slices)

    axes = []
    for i in range(n_row):
        for j in range(n_col):
            ind = i*n_col + j
            # print "Image stack:", ind
            if ind >= n_frames:
                break

            image_stack = image_sequence[frames[ind]]
            label = labels[ind]

            gs = gridspec.GridSpecFromSubplotSpec(
                2, 2,
                width_ratios=[n_w, n_slices],
                height_ratios=[n_h, n_slices],
                subplot_spec=gs_master[i, j],
                wspace=wspace, hspace=hspace)

            ax_z = plt.Subplot(fig, gs[0, 0])
            if label is not None:
                ax_z.set_title(label)
            ax_y = plt.Subplot(fig, gs[1, 0], sharex=ax_z)
            ax_x = plt.Subplot(fig, gs[0, 1], sharey=ax_z)

            project_hdr_stack(
                image_stack, levels=levels,
                fill=fill, threshold=threshold,
                axes=(ax_z, ax_y, ax_x), colors=colors,
                subfigwidth=subfigwidth, **kwargs)

            fig.add_subplot(ax_z)
            fig.add_subplot(ax_y)
            fig.add_subplot(ax_x)

            axes.append((ax_z, ax_y, ax_x))

    return fig, axes


def draw_points_on_stack_projections(
        axes, pos, marker="o", color="white", mew=1, mec="blue",
        projections=["z", "y", "x"], **kwargs):
    """Draw points on three projections.

    Parameters
    ----------
    pos : array
        An array of positions in the image coordinates (slice, y, x).
    axes : tuple
        A tuple of axes for z, y, x projections respectively.
    ...

    """
    ax_z, ax_y, ax_x = [None, None, None]
    for i, method in enumerate(projections):
        if method == "z":
            ax_z = axes[i]
        elif method == "y":
            ax_y = axes[i]
        elif method == "x":
            ax_x = axes[i]
        else:
            raise Exception("Axis label is wrong!")

    pos = np.array(pos).reshape(-1, 3)
    z = pos[:, 0]
    y = pos[:, 1]
    x = pos[:, 2]
    # Note: we remove repetitions for better visualisation:

    if ax_z is not None:
        xy_unique = np.array(
            list(set([tuple([_x, _y]) for _x, _y in zip(x, y)])))
        ax_z.plot(xy_unique[:, 0], xy_unique[:, 1],
                  marker=marker, color=color, linestyle=" ", mew=mew, mec=mec,
                  **kwargs)
    if ax_y is not None:
        xz_unique = np.array(
            list(set([tuple([_x, _z]) for _x, _z in zip(x, z)])))
        ax_y.plot(xz_unique[:, 0], xz_unique[:, 1],
                  marker=marker, color=color, linestyle=" ", mew=mew, mec=mec,
                  **kwargs)
    if ax_x is not None:
        zy_unique = np.array(
            list(set([tuple([_z, _y]) for _z, _y in zip(z, y)])))
        ax_x.plot(zy_unique[:, 0], zy_unique[:, 1],
                  marker=marker, color=color, linestyle=" ", mew=mew, mec=mec,
                  **kwargs)


def draw_line_segment_on_stack_projections(
        axes, pos_start, pos_end, color="blue", lw=1,
        projections=["z", "y", "x"], **kwargs):
    ax_z, ax_y, ax_x = [None, None, None]
    for i, proj in enumerate(projections):
        if proj == "z":
            ax_z = axes[i]
        elif proj == "y":
            ax_y = axes[i]
        elif proj == "x":
            ax_x = axes[i]
        else:
            raise Exception("Axis label is wrong!")
    pos_start = np.array(pos_start)
    pos_end = np.array(pos_end)

    if len(pos_start.shape) == 1:
        pos_start = pos_start.reshape(1, 3)

    if len(pos_end.shape) == 1:
        pos_end = pos_end.reshape(1, 3)

    x = np.concatenate([pos_start[:, 2], pos_end[:, 2]])
    y = np.concatenate([pos_start[:, 1], pos_end[:, 1]])
    z = np.concatenate([pos_start[:, 0], pos_end[:, 0]])

    # Draw line segments:
    if ax_z is not None:
        ax_z.plot(x, y, color=color, linestyle="-", lw=lw, **kwargs)
    if ax_y is not None:
        ax_y.plot(x, z, color=color, linestyle="-", lw=lw, **kwargs)
    if ax_x is not None:
        ax_x.plot(z, y, color=color, linestyle="-", lw=lw, **kwargs)

    # If these are points, draw points:
    if ax_z is not None and np.allclose(*x) and np.allclose(*y):
        ax_z.plot(x, y, color=color, linestyle="", marker=".",
                  ms=2*lw, **kwargs)
    if ax_y is not None and np.allclose(*x) and np.allclose(*z):
        ax_y.plot(x, z, color=color, linestyle="", marker=".",
                  ms=2*lw, **kwargs)
    if ax_x is not None and np.allclose(*z) and np.allclose(*y):
        ax_x.plot(z, y, color=color, linestyle="", marker=".",
                  ms=2*lw, **kwargs)



def draw_box_on_stack_projections(
        axes, bbox, color="blue", alpha=1, lw=1, **kwargs):
    """Box is defined by bottom left and top right corners."""
    bbox = np.array(bbox)

    bl, tr = bbox
    z_min, y_min, x_min = bl
    z_max, y_max, x_max = tr

    p1 = np.array([z_min, y_min, x_min])
    p2 = np.array([z_min, y_min, x_max])
    p3 = np.array([z_min, y_max, x_max])
    p4 = np.array([z_min, y_max, x_min])

    p5 = np.array([z_max, y_min, x_min])
    p6 = np.array([z_max, y_min, x_max])
    p7 = np.array([z_max, y_max, x_max])
    p8 = np.array([z_max, y_max, x_min])

    # -> Top facet:
    draw_line_segment_on_stack_projections(
        axes, p1, p2,
        lw=lw, color=color, alpha=alpha, **kwargs)
    draw_line_segment_on_stack_projections(
        axes, p2, p3,
        lw=lw, color=color, alpha=alpha, **kwargs)
    draw_line_segment_on_stack_projections(
        axes, p3, p4,
        lw=lw, color=color, alpha=alpha, **kwargs)
    draw_line_segment_on_stack_projections(
        axes, p4, p1,
        lw=lw, color=color, alpha=alpha, **kwargs)

    # -> Bottom facet:
    draw_line_segment_on_stack_projections(
        axes, p5, p6,
        lw=lw, color=color, alpha=alpha, **kwargs)
    draw_line_segment_on_stack_projections(
        axes, p6, p7,
        lw=lw, color=color, alpha=alpha, **kwargs)
    draw_line_segment_on_stack_projections(
        axes, p7, p8,
        lw=lw, color=color, alpha=alpha, **kwargs)
    draw_line_segment_on_stack_projections(
        axes, p8, p5,
        lw=lw, color=color, alpha=alpha, **kwargs)

    # -> Connections:
    draw_line_segment_on_stack_projections(
        axes, p1, p5,
        lw=lw, color=color, alpha=alpha, **kwargs)
    draw_line_segment_on_stack_projections(
        axes, p2, p6,
        lw=lw, color=color, alpha=alpha, **kwargs)
    draw_line_segment_on_stack_projections(
        axes, p3, p7,
        lw=lw, color=color, alpha=alpha, **kwargs)
    draw_line_segment_on_stack_projections(
        axes, p4, p8,
        lw=lw, color=color, alpha=alpha, **kwargs)


def draw_circle_on_stack_projections(
        axes, center, radius, color="blue", alpha=1., fill=False,
        zorder=100, lw=1, **kwargs):

    """Draw circle with same radius on three projections.

    Parameters
    ----------
    center : array
        An array of centers in image space (slice, y, x).
    radius : array
        An array of radius for each sphere in image space (r). If scolar, then
        then it is a radius of all spheres.
    axes : tuple
        A tuple of axes for z, y, x projections respectively.
    ...

    """
    lw = 0 if fill is True else lw

    ax_z, ax_y, ax_x = axes

    # Adjust the list of centers:
    center = np.array(center)
    if len(center.shape) == 1:
        center = center.reshape(1, 3)
    n_centers = len(center)

    # Adjust the list of radii for each sphere:
    radius = (
        np.array(radius).reshape(-1)
        if n_centers == np.array(radius).size else
        [radius]*n_centers)

    center_z = center[:, 0]
    center_y = center[:, 1]
    center_x = center[:, 2]

    for x, y, z, r in zip(center_x, center_y, center_z, radius):
        circle_z = plt.Circle(
            (x, y), r,
            color=color, alpha=alpha, fill=fill, clip_on=True, lw=lw,
            zorder=zorder, **kwargs)
        ax_z.add_artist(circle_z)

        circle_y = plt.Circle(
            (x, z), r,
            color=color, alpha=alpha, fill=fill, clip_on=True, lw=lw,
            zorder=zorder, **kwargs)
        ax_y.add_artist(circle_y)

        circle_x = plt.Circle(
            (z, y), r,
            color=color, alpha=alpha, fill=fill, clip_on=True, lw=lw,
            zorder=zorder, **kwargs)
        ax_x.add_artist(circle_x)


def draw_ellipse_on_stack_projections(
        axes, center, radius, color="blue", alpha=1., fill=False,
        zorder=100, lw=1, **kwargs):

    """Draw ellipse on three projections.

    Parameters
    ----------
    center : array
        An array of centers in image space (slice, y, x).
    radius : array
        An array of radii in image space (rz, ry, rx).
    axes : tuple
        A tuple of axes for z, y, x projections respectively.
    ...

    """
    ax_z, ax_y, ax_x = axes
    center = np.array(center)
    n_centers = len(center)

    radius = np.array(radius)

    # Adjust the list of centers:
    center = np.array(center)
    if len(center.shape) == 1:
        center = center.reshape(1, 3)
    n_centers = len(center)

    # Adjust the list of radii for each ellipsoid:
    if (len(radius.shape) == 1) & (n_centers == 1):
        radius = radius.reshape(1, 3)
    elif len(radius.shape) == n_centers:
        radius = np.tile(radius, (n_centers, 1))

    diameter_is = 2.0*radius

    center_x = center[:, 2]
    center_y = center[:, 1]
    center_z = center[:, 0]

    for x, y, z, d in zip(center_x, center_y, center_z, diameter_is):
        dz, dy, dx = d

        ellipse_z = mpatch.Ellipse(
            xy=(x, y), width=dx, height=dy,
            color=color, alpha=alpha, fill=fill, clip_on=True, lw=lw,
            zorder=zorder, **kwargs)
        ax_z.add_patch(ellipse_z)

        ellipse_y = mpatch.Ellipse(
            xy=(x, z), width=dx, height=dz,
            color=color, alpha=alpha, fill=fill, clip_on=True, lw=lw,
            zorder=zorder, **kwargs)
        ax_y.add_patch(ellipse_y)

        ellipse_x = mpatch.Ellipse(
            xy=(z, y), width=dz, height=dy,
            color=color, alpha=alpha, fill=fill, clip_on=True, lw=lw,
            zorder=zorder, **kwargs)
        ax_x.add_patch(ellipse_x)


def outline_pixels(ax, mask, color="green", **kwargs):
    """

    Parameters
    ----------
    ax : axis
        An axis of a figure where pixels have to be outlined.
    mask : array
        A 2 dimensional binary mask of pixels which have to be outlined.

    """
    assert len(mask.shape) == 2
    h, w = mask.shape
    u_grid, v_grid = np.meshgrid(range(h), range(w), indexing="ij")

    # Get pixel vertices in pixel grid coordinates:
    tl, tr, br, bl = np.zeros((4, h, w, 2))
    tl[:, :, 0], tl[:, :, 1] = np.array(u_grid - 0.5), np.array(v_grid - 0.5)
    tr[:, :, 0], tr[:, :, 1] = np.array(u_grid - 0.5), np.array(v_grid + 0.5)
    br[:, :, 0], br[:, :, 1] = np.array(u_grid + 0.5), np.array(v_grid + 0.5)
    bl[:, :, 0], bl[:, :, 1] = np.array(u_grid + 0.5), np.array(v_grid - 0.5)

    def _get_px_bound(s, m):
        return np.logical_and(s, np.logical_xor(m, s))

    bshift = np.roll(mask, 1, axis=0)
    bshift[0, :] = 0

    tshift = np.roll(mask, -1, axis=0)
    tshift[-1, :] = 0

    rshift = np.roll(mask, 1, axis=1)
    rshift[:, 0] = 0

    lshift = np.roll(mask, -1, axis=1)
    lshift[:, -1] = 0

    top = _get_px_bound(bshift, mask)
    bottom = _get_px_bound(tshift, mask)
    left = _get_px_bound(rshift, mask)
    right = _get_px_bound(lshift, mask)

    edges_list = []
    for u, v in zip(u_grid.flatten(), v_grid.flatten()):
        if bottom[u, v]:
            edges_list.append([bl[u, v, :], br[u, v, :]])
        if top[u, v]:
            edges_list.append([tl[u, v, :], tr[u, v, :]])
        if left[u, v]:
            edges_list.append([tl[u, v, :], bl[u, v, :]])
        if right[u, v]:
            edges_list.append([tr[u, v, :], br[u, v, :]])
    edges = np.array(edges_list)

    for edge in edges:
        ax.plot(edge[:, 1], edge[:, 0], color=color, **kwargs)


def outline_pixels_on_stack_projections(
        axes, mask, color="blue", **kwargs):
    """Draw pixel outlines for 3d projection.
    axes : tuple
        A tuple of axes for z, y, x projections respectively.
    mask : array
        A 3 dimensional binary mask of pixels which have to be outlined.
    """
    mask_zproj = project_along_axis(mask, 0, "max")
    outline_pixels(axes[0], mask_zproj, color, **kwargs)
    mask_yrpoj = project_along_axis(mask, 1, "max")
    outline_pixels(axes[1], mask_yrpoj, color, **kwargs)
    mask_xrpoj = project_along_axis(mask, 2, "max")
    outline_pixels(axes[2], mask_xrpoj.T, color, **kwargs)
