import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.axes_grid1 import make_axes_locatable

from plotting_parameters import *

def latexify(fig_width=None, fig_height=None, height_mult=1, width_mult=1, columns=1, square=False):
    """Set up matplotlib's RC params for LaTeX plotting.
    Call this before plotting a figure.

    Parameters
    ----------
    fig_width : float, optional, inches
    fig_height : float,  optional, inches
    columns : {1, 2}, optional
    square: boolean,optional
    """

    # code adapted from http://www.scipy.org/Cookbook/Matplotlib/LaTeX_Examples

    # Width and max height in inches for IEEE journals taken from
    # computer.org/cms/Computer.org/Journal%20templates/transactions_art_guide.pdf

    assert(columns in [1,2,3,4])

    if fig_width is None:
        if columns == 1:
            fig_width = COLUMN_WIDTH
        elif columns == 2:
            fig_width = COLUMN_WIDTH * COLUMN_HALFSIZE  
        elif columns == 3:
            fig_width = COLUMN_WIDTH * COLUMN_THIRDSIZE
        elif columns == 4:
            fig_width = COLUMN_WIDTH * 0.24

    fig_width *= width_mult

    if fig_height is None:
        golden_mean = (math.sqrt(5)-1.0)/2.0    # Aesthetic ratio
        fig_height = fig_width*golden_mean # height in inches

    fig_height *= height_mult
        
    if square:
        fig_height = fig_width
        
    MAX_HEIGHT_INCHES = 8.0
    if fig_height > MAX_HEIGHT_INCHES:
        print("WARNING: fig_height too large:" + fig_height + 
              "so will reduce to" + MAX_HEIGHT_INCHES + "inches.")
        fig_height = MAX_HEIGHT_INCHES

    params = {'backend': 'ps',
              'text.latex.preamble': '\\usepackage{gensymb}\n\\usepackage{amsmath}',
              'axes.labelsize': FONTSIZE, # fontsize for x and y labels (was 10)
              'axes.titlesize': FONTSIZE,
              'font.size': FONTSIZE, # was 10
              'legend.fontsize': FONTSIZE, # was 10
              'xtick.labelsize': FONTSIZE,
              'ytick.labelsize': FONTSIZE,
              'lines.linewidth': 1.0,
              'text.usetex': True,
              'figure.figsize': [fig_width,fig_height],
              'font.family': 'serif'
    }

    matplotlib.rcParams.update(params)

def remove_spines(axis, axis_side='left', sharex=False, sharey=False):
    axis.spines['top'].set_visible(False)
    if not sharex:
        axis.xaxis.set_ticks_position('bottom')
    if axis_side == 'left':
        axis.spines['right'].set_visible(False)
        if not sharey:
            axis.yaxis.set_ticks_position('left')
            axis.yaxis.set_label_position('left')
    else:
        axis.spines['left'].set_visible(False)
        if not sharey:
            axis.yaxis.set_ticks_position('right')
            axis.yaxis.set_label_position('right')

def create_axes(n_columns=1, axis_side='left', height_mult=1, width_mult=1, subplots_rows=1, subplots_columns=1, sharex=False, sharey=False, square=False):
    latexify(columns=n_columns, square=square, height_mult=height_mult, width_mult=width_mult)
    fig, axes = plt.subplots(subplots_rows, subplots_columns, sharex=sharex, sharey=sharey)
    if subplots_rows == 1 and subplots_columns == 1:
        remove_spines(axes, axis_side)
    else:
        for axis in axes.flatten():
            remove_spines(axis, axis_side, sharex=sharex, sharey=sharey)
    return fig, axes

def save_plot(filename):
    plt.savefig(filename, pad_inches=PAD_INCHES, bbox_inches = 'tight')
    plt.show()

def attach_colorbar(axis, im, side='right'):
    divider = make_axes_locatable(axis)
    cax = divider.append_axes(side, size="5%", pad=0.05)
    if side == 'right' or side =='left':
        orientation = 'vertical'
    else:
        orientation = 'horizontal'
    return plt.colorbar(im, cax=cax, orientation=orientation)
