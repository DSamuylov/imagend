"""A small python package to visualize 3 and 4 dimensional image data.

Author: Denis Samuylov, Prateek Purwar
E-mail: denis.samuylov@gmail.com
        pp.semtex@gmail.com

"""

import math
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as font_manager
import matplotlib.patches as mpatch


class DimensionError(Exception):

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr("The image has a wrong dimension ({})".format(self.value))


def project(img, axis, proj='mean'):
    """Mean|Median|Max projection of an image along a given axis"""
    assert proj in ['mean', 'median', 'max']
    if proj == 'mean':
        return np.mean(img, axis=axis)
    elif proj == 'median':
        return np.median(img, axis=axis)
    elif proj == 'max':
        return np.max(img, axis=axis)
    else:
        raise Exception("Wrong projection!")


def compute_project_normalization(img_stack_list, proj="mean"):
    proj_list = [project(img_stack, ax_id, proj)
                 for img_stack in img_stack_list for ax_id in range(3)]
    vmin = min([_proj.min() for _proj in proj_list])
    vmax = max([_proj.max() for _proj in proj_list])
    return vmin, vmax


def project_image_stack(img, vmin=None, vmax=None, axes=None,
                        proj="mean", figwidth=3,
                        cmap=cm.magma_r, alpha=1, aspect=1,
                        projections=["z", "y", "x"]):
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
    n_s, n_h, n_w = img.shape

    z_proj = project(img, axis=0, proj=proj)
    y_proj = project(img, axis=1, proj=proj)
    x_proj = project(img, axis=2, proj=proj).T

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
        for i, proj in enumerate(projections):
            if proj == "z":
                ax_z = axes[i]
            elif proj == "y":
                ax_y = axes[i]
            elif proj == "x":
                ax_x = axes[i]
            else:
                raise Exception("Axis label is wrong!")

    if "z" in projections:
        ax_z.imshow(z_proj, interpolation='nearest', cmap=cmap, aspect=1,
                    vmin=vmin, vmax=vmax, alpha=alpha)
        ax_z.set_xlim([-0.5, n_w - 0.5])
        ax_z.set_ylim([n_h - 0.5, -0.5])
        ax_z.axis('off')
    if "y" in projections:
        ax_y.imshow(y_proj, interpolation='nearest', cmap=cmap, aspect=1,
                    vmin=vmin, vmax=vmax, alpha=alpha)
        ax_y.set_xlim([-0.5, n_w-0.5])
        ax_y.set_ylim([n_s-0.5, -0.5])
        ax_y.axis('off')
        ax_y.set_aspect(aspect)
    if "x" in projections:
        ax_x.imshow(x_proj, interpolation='nearest', cmap=cmap, aspect=1,
                    vmin=vmin, vmax=vmax, alpha=alpha)
        ax_x.set_xlim([-0.5, n_s-0.5])
        ax_x.set_ylim([n_h-0.5, -0.5])
        ax_x.axis('off')
        ax_x.set_aspect(1/aspect)

    # if add_labels:
    #     # Draw xyz labels:
    #     font = font_manager.FontProperties()
    #     # font.set_weight('bold')
    #     font.set_size(15)
    #     ax_z.text(2, 4, 'z', color='white', fontproperties=font)
    #     ax_y.text(2, 4, 'y', color='white', fontproperties=font)
    #     ax_x.text(2, 4, 'x', color='white', fontproperties=font)
    axes = []
    for proj in projections:
        if proj == "z":
            axes.append(ax_z)
        if proj == "y":
            axes.append(ax_y)
        if proj == "x":
            axes.append(ax_x)
    return fig, axes


def project_image_sequence(img_sequence, frames=None,
                           nsubfig=None, subfigwidth=3, ncol=5,
                           vmin=None, vmax=None, proj='mean',
                           add_labels=False, cmap=cm.magma_r, add_title=True):
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
    elif isinstance(frames, int):
        frames = np.linspace(0, n_frames, frames,
                             endpoint=False, dtype=np.uint8)

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
                wspace=wspace, hspace=hspace)

            ax_z = plt.Subplot(fig, gs[0, 0])
            if add_title is True:
                ax_z.set_title('frame={}'.format(frames[ind]))
            ax_y = plt.Subplot(fig, gs[1, 0], sharex=ax_z)
            ax_x = plt.Subplot(fig, gs[0, 1], sharey=ax_z)

            project_image_stack(img_sequence[frames[ind]],
                          axes=[ax_z, ax_y, ax_x],
                          vmin=vmin, vmax=vmax, proj=proj, cmap=cmap)
            fig.add_subplot(ax_z)
            fig.add_subplot(ax_y)
            fig.add_subplot(ax_x)
            axes.append((ax_z, ax_y, ax_x))
    return fig, axes


def _project_hdr(mask, axes=None, label=None, figwidth=3, alpha=1, aspect=1,
                 fill=False, fontname="Times New Roman", fontsize=10,
                 projections=["z", "y", "x"], **kwargs):
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
        if ("z" in projections) & ("x" in projections):
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
        for i, proj in enumerate(projections):
            if proj == "z":
                ax_z = axes[i]
            elif proj == "y":
                ax_y = axes[i]
            elif proj == "x":
                ax_x = axes[i]
            else:
                raise Exception("Axis label is wrong!")

    # if fill is True:
    #     z_proj = np.max(mask, axis=0)
    #     y_proj = np.max(mask, axis=1)
    #     x_proj = np.swapaxes(np.max(mask, axis=2), 0, 1)

    #     ax_z.imshow(z_proj, interpolation="nearest")
    #     ax_y.imshow(y_proj, interpolation="nearest")
    #     ax_x.imshow(x_proj, interpolation="nearest")

    if fill is False:

        mask_empty = np.zeros((n_s, n_h, n_w, 4))
        if ax_z is not None:
            ax_z.imshow(project(mask_empty, axis=0))
        if ax_y is not None:
            ax_y.imshow(project(mask_empty, axis=1))
        if ax_x is not None:
            ax_z.imshow(np.swapaxes(project(mask_empty, axis=2), 0, 1))
        # Note: we project an empy mask to avoid problems with the origin.

        _y = np.arange(n_h)
        _x = np.arange(n_w)
        _z = np.arange(n_s)

        _mask = mask.max(axis=-1) > 0
        _color_rgba = mask.reshape(-1, 4).max(axis=0)
        _rgb, _alpha = _color_rgba[0:3], _color_rgba[3]

        if ax_z is not None:
            y_list, x_list = np.meshgrid(_y, _x, indexing='ij')
            z_proj = np.zeros((n_h, n_w), np.uint8)
            z_proj[np.max(_mask, axis=0)] = 1
            ax_z.contour(
                x_list, y_list, z_proj,
                levels=[0, 1], colors=[_rgb], alpha=_alpha, origin="image",
                **kwargs)

        if ax_y is not None:
            z_list, x_list = np.meshgrid(_z, _x, indexing='ij')
            y_proj = np.zeros((n_s, n_w), np.uint8)
            y_proj[np.max(_mask, axis=1)] = 1
            ax_y.contour(
                x_list, z_list, y_proj,
                levels=[0, 1], colors=[_rgb], alpha=_alpha, origin="lower",
                **kwargs)

        if ax_x is not None:
            x_list, z_list = np.meshgrid(_y, _z, indexing='ij')
            x_proj = np.zeros((n_h, n_s), np.uint8)
            x_proj[np.swapaxes(np.max(_mask, axis=2), 0, 1)] = 1
            ax_x.contour(
                z_list, x_list, x_proj,
                levels=[0, 1], colors=[_rgb], alpha=_alpha, origin="lower",
                **kwargs)

    if ax_z is not None:
        if label is not None:
            ax_z.set_title(label, fontname=fontname, fontsize=fontsize)
        ax_z.set_xlim([-0.5, n_w - 0.5])
        ax_z.set_ylim([n_h - 0.5, -0.5])
        ax_z.axis('off')
        ax_z.set_aspect(aspect)
    if ax_y is not None:
        ax_y.set_xlim([-0.5, n_w-0.5])
        ax_y.set_ylim([n_s-0.5, -0.5])
        ax_y.axis('off')
        ax_y.set_aspect(aspect)
    if ax_x is not None:
        ax_x.set_xlim([-0.5, n_s-0.5])
        ax_x.set_ylim([n_h-0.5, -0.5])
        ax_x.axis('off')
        ax_x.set_aspect(aspect)

    axes = []
    for proj in projections:
        if proj == "z":
            axes.append(ax_z)
        if proj == "y":
            axes.append(ax_y)
        if proj == "x":
            axes.append(ax_x)
    return fig, axes


def project_hdr_stack(img_stack, levels=[0.95, 0.99], label=None, axes=None,
                      cmap=cm.magma_r, colors=None,
                      fill=False, threshold=10e-10,
                      fontname="Times New Roman", fontsize=10,
                      figwidth=3, projections=["z", "y", "x"],
                      **kwargs):
    """
    References: Computing and Graphing Highest Density Regions
    """
    levels = sorted(levels)
    n_levels = len(levels)
    if colors is None:
        colors = cmap(np.linspace(0, 1, n_levels))

    print "Fill hdr: {}".format(fill)
    print "Range of img_stack: {} .. {}".format(
        img_stack.min(), img_stack.max())
    print "Threshold value: {}".format(threshold)

    _img_stack = img_stack.copy()
    indexes = np.where(_img_stack > _img_stack.min() + threshold)
    vals = _img_stack[indexes]
    print "Number of threshold pixels: {} (out of {})".format(
        vals.size, img_stack.size)
    n_px = vals.size

    f_alpha_list = np.unique(vals)
    p_list = np.zeros(f_alpha_list.size)
    for i, f_alpha in enumerate(f_alpha_list):
        p_list[i] = np.sum(img_stack >= f_alpha, dtype=np.float64)/n_px
    alpha_list = 1. - p_list

    hdr_list = np.zeros((n_levels,) + img_stack.shape + (4,))
    for level_id, level in enumerate(levels):
        i = np.squeeze(np.where(alpha_list >= level))[0]
        f_alpha = f_alpha_list[i]
        mask = img_stack >= f_alpha
        hdr_list[level_id, mask > 0] = colors[level_id]
    #     print "Level {}%: #pixels={}/{} ({:.2f}%)".format(
    #         level*100.,
    #         np.sum(mask), mask.size,
    #         np.sum(mask, dtype=np.float64)/mask.size*100.)
    # print "\n"

    for i, hdr in enumerate(hdr_list):
        _fig, axes = _project_hdr(
            hdr, axes=axes, label=label, fill=fill, figwidth=figwidth,
            fontname=fontname, fontsize=fontsize, projections=projections,
            **kwargs)
        if i == 0:
            fig = _fig
    return fig, axes


def compare_hdr_stack_projections(
        img_stacks, levels=[0.95, 0.99], fill=False, threshold=10e-10,
        labels=None, fontname="Times New Roman", fontsize=10,
        cmap=cm.magma_r, colors=None, subfigwidth=3, **kwargs):

    img_stacks = np.array(img_stacks)
    n_imgs, n_slices, n_h, n_w = img_stacks.shape

    if labels is None:
        # labels = ["image {}".format(i) for i in range(n_imgs)]
        labels = [None]*n_imgs

    n_col = n_imgs
    n_row = 1

    subfigheight = (subfigwidth)*np.float64(n_h + n_slices)/(n_w + n_slices)

    fig = plt.figure(figsize=(subfigwidth*n_col, subfigheight*(n_row)))
    gs_master = gridspec.GridSpec(n_row, n_col)

    axes = []
    for i in range(n_row):
        for j in range(n_col):
            ind = i*n_col + j
            img_stack = img_stacks[ind]
            label = labels[ind]

            gs = gridspec.GridSpecFromSubplotSpec(
                2, 2,
                width_ratios=[n_w, n_slices],
                height_ratios=[n_h, n_slices],
                subplot_spec=gs_master[i, j],
                wspace=0.05, hspace=0.05)

            ax_z = plt.Subplot(fig, gs[0, 0])
            if label is not None:
                ax_z.set_title(label, fontname=fontname, fontsize=fontsize)
            ax_y = plt.Subplot(fig, gs[1, 0], sharex=ax_z)
            ax_x = plt.Subplot(fig, gs[0, 1], sharey=ax_z)

            project_hdr_stack(img_stack, levels=levels,
                              fill=fill, threshold=threshold,
                              fontname=fontname, fontsize=fontsize,
                              axes=(ax_z, ax_y, ax_x), colors=colors,
                              subfigwidth=subfigwidth, cmap=cmap, **kwargs)

            fig.add_subplot(ax_z)
            fig.add_subplot(ax_y)
            fig.add_subplot(ax_x)

            axes.append((ax_z, ax_y, ax_x))

    return fig, axes


def draw_points_in_stack_projections(axes, pos_is, marker='.', color='r',
                                     mew=0, projections=["z", "y", "x"],
                                     **kwargs):
    """Draw points on three projections.

    Parameters
    ----------
    pos_is : array
        An array of positions in image space (slice, y, x).
    axes : tuple
        A tuple of axes for z, y, x projections respectively.
    ...

    """
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

    pos_is = np.array(pos_is).reshape(-1, 3)
    z = pos_is[:, 0]
    y = pos_is[:, 1]
    x = pos_is[:, 2]
    # Note: we remove repetitions for better visualisation:

    if ax_z is not None:
        xy_unique = np.array(
            list(set([tuple([_x, _y]) for _x, _y in zip(x, y)])))
        ax_z.plot(xy_unique[:, 0], xy_unique[:, 1],
                  marker=marker, color=color, linestyle=' ', mew=mew, **kwargs)
    if ax_y is not None:
        xz_unique = np.array(
            list(set([tuple([_x, _z]) for _x, _z in zip(x, z)])))
        ax_y.plot(xz_unique[:, 0], xz_unique[:, 1],
                  marker=marker, color=color, linestyle=' ', mew=mew, **kwargs)
    if ax_x is not None:
        zy_unique = np.array(
            list(set([tuple([_z, _y]) for _z, _y in zip(z, y)])))
        ax_x.plot(zy_unique[:, 0], zy_unique[:, 1],
                  marker=marker, color=color, linestyle=' ', mew=mew, **kwargs)


def draw_line_segment_in_stack_projections(
        axes, pos_is_start, pos_is_end, color="red",
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

    if len(pos_is_start.shape) == 1:
        pos_is_start = pos_is_start.reshape(1, 3)

    if len(pos_is_end.shape) == 1:
        pos_is_end = pos_is_end.reshape(1, 3)

    x = np.concatenate([pos_is_start[:, 2], pos_is_end[:, 2]])
    y = np.concatenate([pos_is_start[:, 1], pos_is_end[:, 1]])
    z = np.concatenate([pos_is_start[:, 0], pos_is_end[:, 0]])

    if ax_z is not None:
        ax_z.plot(x, y, color=color, linestyle='-', **kwargs)
    if ax_y is not None:
        ax_y.plot(x, z, color=color, linestyle='-', **kwargs)
    if ax_x is not None:
        ax_x.plot(z, y, color=color, linestyle='-', **kwargs)


def draw_box_in_stack_projections(axes, bbox, color="red", alpha=0.5,
                                  ls=3, **kwargs):
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

    # p1 = bbox[0, 0, 0].reshape(1, 3)
    # p2 = bbox[-1, 0, 0].reshape(1, 3)
    # p3 = bbox[-1, -1, 0].reshape(1, 3)
    # p4 = bbox[0, -1, 0].reshape(1, 3)

    # p5 = bbox[0, 0, -1].reshape(1, 3)
    # p6 = bbox[-1, 0, -1].reshape(1, 3)
    # p7 = bbox[-1, -1, -1].reshape(1, 3)
    # p8 = bbox[0, -1, -1].reshape(1, 3)

    # -> Top facet:
    draw_line_segment_in_stack_projections(
        axes, p1, p2,
        lw=ls, color=color, alpha=alpha, **kwargs)
    draw_line_segment_in_stack_projections(
        axes, p2, p3,
        lw=ls, color=color, alpha=alpha, **kwargs)
    draw_line_segment_in_stack_projections(
        axes, p3, p4,
        lw=ls, color=color, alpha=alpha, **kwargs)
    draw_line_segment_in_stack_projections(
        axes, p4, p1,
        lw=ls, color=color, alpha=alpha, **kwargs)

    # -> Bottom facet:
    draw_line_segment_in_stack_projections(
        axes, p5, p6,
        lw=ls, color=color, alpha=alpha, **kwargs)
    draw_line_segment_in_stack_projections(
        axes, p6, p7,
        lw=ls, color=color, alpha=alpha, **kwargs)
    draw_line_segment_in_stack_projections(
        axes, p7, p8,
        lw=ls, color=color, alpha=alpha, **kwargs)
    draw_line_segment_in_stack_projections(
        axes, p8, p5,
        lw=ls, color=color, alpha=alpha, **kwargs)

    # -> Connections:
    draw_line_segment_in_stack_projections(
        axes, p1, p5,
        lw=ls, color=color, alpha=alpha, **kwargs)
    draw_line_segment_in_stack_projections(
        axes, p2, p6,
        lw=ls, color=color, alpha=alpha, **kwargs)
    draw_line_segment_in_stack_projections(
        axes, p3, p7,
        lw=ls, color=color, alpha=alpha, **kwargs)
    draw_line_segment_in_stack_projections(
        axes, p4, p8,
        lw=ls, color=color, alpha=alpha, **kwargs)


def draw_circle_in_stack_projections(axes, center_is, radius_is,
                                     color="red", alpha=1., fill=False,
                                     zorder=100, lw=2, **kwargs):

    """Draw circle with same radius on three projections.

    Parameters
    ----------
    center_is : array
        An array of centers in image space (slice, y, x).
    radius_is : array
        An array of radius for each sphere in image space (r). If scolar, then
        then it is a radius of all spheres.
    axes : tuple
        A tuple of axes for z, y, x projections respectively.
    ...

    """

    lw = 0 if fill is True else lw

    ax_z, ax_y, ax_x = axes

    # Adjust the list of centers:
    center_is = np.array(center_is)
    if len(center_is.shape) == 1:
        center_is = center_is.reshape(1, 3)
    n_centers = len(center_is)

    # Adjust the list of radii for each sphere:
    radius_is = (
        np.array(radius_is)
        if n_centers == np.array(radius_is).size else
        [radius_is]*n_centers)

    center_z = center_is[:, 0]
    center_y = center_is[:, 1]
    center_x = center_is[:, 2]

    for x, y, z, r in zip(center_x, center_y, center_z, radius_is):
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


def draw_ellipse_in_stack_projections(axes, center_is, radius_is,
                                      color="red", alpha=1., fill=False,
                                      zorder=100, lw=2, **kwargs):

    """Draw ellipse on three projections.

    Parameters
    ----------
    center_is : array
        An array of centers in image space (slice, y, x).
    radius_is : array
        An array of radii in image space (rz, ry, rx).
    axes : tuple
        A tuple of axes for z, y, x projections respectively.
    ...

    """
    ax_z, ax_y, ax_x = axes
    center_is = np.array(center_is)
    n_centers = len(center_is)

    radius_is = np.array(radius_is)

    # Adjust the list of centers:
    center_is = np.array(center_is)
    if len(center_is.shape) == 1:
        center_is = center_is.reshape(1, 3)
    n_centers = len(center_is)

    # Adjust the list of radii for each ellipsoid:
    if (len(radius_is.shape) == 1) & (n_centers == 1):
        radius_is = radius_is.reshape(1, 3)
    elif len(radius_is.shape) == n_centers:
        radius_is = np.tile(radius_is, (n_centers, 1))

    diameter_is = 2.0*radius_is

    center_x = center_is[:, 2]
    center_y = center_is[:, 1]
    center_z = center_is[:, 0]

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


def draw_pixel_outlines(ax, mask, color='green', **kwargs):
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
    u_grid, v_grid = np.meshgrid(range(h), range(w), indexing='ij')

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


def draw_pixel_outlines_in_stack_projections(axes, mask,
                                             color='green', **kwargs):
    """Draw pixel outlines for 3d projection.
    axes : tuple
        A tuple of axes for z, y, x projections respectively.
    mask : array
        A 3 dimensional binary mask of pixels which have to be outlined.
    """
    mask_zproj = project(mask, 0, 'max')
    draw_pixel_outlines(axes[0], mask_zproj, color, **kwargs)
    mask_yrpoj = project(mask, 1, 'max')
    draw_pixel_outlines(axes[1], mask_yrpoj, color, **kwargs)
    mask_xrpoj = project(mask, 2, 'max')
    draw_pixel_outlines(axes[2], mask_xrpoj.T, color, **kwargs)


def compare_stack_projections(imgs, labels=None, subfigwidth=3,
                              vmin=None, vmax=None, proj='mean',
                              drop_y_proj=False, add_labels=False,
                              normalized=False, cmap=cm.magma_r,
                              fontname="Times New Roman", fontsize=10,
                              **kwargs):
    """Compare projections of 3D stacks.

    Parameters
    ----------
    imgs : array
        A list of 3D stacks.
    labels : list
        List of labels for each image.
    subfigwidth : scalar
        The figure width is in inches. The height is computed automatically to
        preserve the aspect ratio.
    vmin, vmax : scalar, optional, default: None
        vmin and vmax are used to normalize luminance data. By default each
        image is normalised to take all range.
    proj : str, optional, default : 'mean'
        It defines the type of projection that we would like to plot. It can
        be: [mean|median|max]
    normalize : boolean, default : False
        If True, normalise all images on their maximum.

    Returns
    -------
    fig : figure
        A created figure.
    axes : list
        A list of tuples with axes for each projection.

    """
    imgs = np.array(imgs)
    n_imgs, n_slices, n_h, n_w = imgs.shape

    if (normalized is not None) and (normalized is not False):
        vmin_list = []
        vmax_list = []
        for img in imgs:
            projected_list = [project(img, ax, proj) for ax in range(3)]
            vmin_list.append(
                min([projected.min() for projected in projected_list]))
            vmax_list.append(
                max([projected.max() for projected in projected_list]))
        if normalized is True:
            vmin = min(vmin_list)
            vmax = max(vmax_list)
        else:
            vmin = vmin_list[normalized]
            vmax = vmax_list[normalized]
        print "Image projections are normalized between: [{}, {}]"\
            .format(vmin, vmax)

    if labels is None:
        # labels = ["image {}".format(i) for i in range(n_imgs)]
        labels = [None]*n_imgs

    n_col = n_imgs
    n_row = 1

    subfigheight = (subfigwidth)*np.float64(n_h + n_slices)/(n_w + n_slices)

    fig = plt.figure(figsize=(subfigwidth*n_col, subfigheight*(n_row)))
    gs_master = gridspec.GridSpec(n_row, n_col)

    axes = []
    for i in range(n_row):
        for j in range(n_col):
            ind = i*n_col + j
            img = imgs[ind]
            label = labels[ind]

            gs = gridspec.GridSpecFromSubplotSpec(
                2, 2,
                width_ratios=[n_w, n_slices],
                height_ratios=[n_h, n_slices],
                subplot_spec=gs_master[i, j],
                wspace=0.05, hspace=0.05)

            ax_z = plt.Subplot(fig, gs[0, 0])
            if label is not None:
                ax_z.set_title(label, fontname=fontname, fontsize=fontsize)
            ax_y = plt.Subplot(fig, gs[1, 0], sharex=ax_z)
            ax_x = plt.Subplot(fig, gs[0, 1], sharey=ax_z)

            project_image_stack(img, vmin=vmin, vmax=vmax,
                                axes=(ax_z, ax_y, ax_x),
                                proj=proj, figwidth=subfigwidth,
                                cmap=cmap, **kwargs)

            fig.add_subplot(ax_z)
            fig.add_subplot(ax_y)
            fig.add_subplot(ax_x)

            axes.append((ax_z, ax_y, ax_x))

    return fig, axes


def show_stack(img_stack, slices_to_plot=None, labels=None, n_cols=5,
               subfigwidth=2, vmin=None, vmax=None, normalized=True,
               title=True, cmap=cm.magma_r,
               fontname="Times New Roman", fontsize=10):
    """Display slices of 3D stack of images"""
    assert len(img_stack.shape) == 3
    n_slices, n_height, n_width = img_stack.shape

    if slices_to_plot is None:
        slices_to_plot = np.arange(n_slices)

    if type(slices_to_plot) == int:
        slices_to_plot = np.linspace(0, n_slices - 1, slices_to_plot,
                                     endpoint=True, dtype=np.uint8)

    # Number of subfigures:
    n_images = len(slices_to_plot)
    n_rows = int(np.ceil(float(n_images)/n_cols))

    # Gray value range:
    vmin = img_stack.min() if normalized & (vmin is None) else vmin
    vmax = img_stack.max() if normalized & (vmax is None) else vmax

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

        ax.axis('off')

        if index < n_images:
            slice_id = slices_to_plot[index]
            image = img_stack[slice_id]
            ax.imshow(image, interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
            if title is True:
                if labels is None:
                    label = 'Slice: {}'.format(slice_id)
                else:
                    label = labels[index]
                ax.set_title(label, fontname=fontname, fontsize=fontsize)

    # fig.tight_layout()
    # Note: if you tight_layout for one slice it adds padding.
    if n_images == 1:
        fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
        # fig.adjust(
        #     hspace = 0, wspace = 0,
        #               bottom=0, top=1, left=0, right=1)
    return fig, axes
