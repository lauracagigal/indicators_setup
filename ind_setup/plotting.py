import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as sp

from .colors import get_default_line_colors, plotting_style

line_colors = get_default_line_colors()
plotting_style()

def plot_trendline(data, var, ax, color = 'k'):
    
    """
    Plots a trendline on a given axis based on the provided data and variable.

    Parameters:
    - data (pandas.DataFrame): The data containing the variable to plot.
    - var (str): The name of the variable to plot.
    - ax (matplotlib.axes.Axes): The axis on which to plot the trendline.
    """

    data = data[var].dropna()

    # Convert time to numerical values (ordinal format)
    time = data.index.values
    time_num = time.view('int64') // 1e9 

    # Fit a trendline (degree 1 polynomial)
    coefficients = np.polyfit(time_num, data.values, 1)  # Linear fit
    trendline = np.poly1d(coefficients)  # Create trendline function

    ax.plot(time, trendline(time_num), linewidth = 3, zorder = 2, c = color, linestyle = ':')


def plot_timeseries(dict_plot, trendline=False):

    """
    Plots lines based on the given dictionary of data.

    Parameters:
    dict_plot (list): A list of dictionaries containing the data and settings for each line plot.
                      Each dictionary should have the following keys:
                      - 'ax': The axis number (1 or 2) on which to plot the line.
                      - 'data': The data to be plotted.
                      - 'var': The variable to be plotted.
                      - 'label': The label for the line plot.
    trendline (bool, optional): Whether to plot a trendline for each line plot. Defaults to False.

    Returns:
    None
    """
    fig, ax = plt.subplots(1, 1, figsize=(15, 6))
    ax2 = ax.twinx()
    ax.tick_params(axis='both', labelsize=14)
    ax2.tick_params(axis='both', labelsize=14)

    lns, c = [], 0
    for entry in dict_plot:
        
        if entry['ax'] == 1:
            ax_plot = ax
        elif entry['ax'] == 2:
            ax_plot = ax2

        l1 = ax_plot.plot(entry['data'].index, entry['data'][entry['var']], label=entry['label'], color=line_colors[c])
        lns.extend(l1)

        if trendline:
            plot_trendline(entry['data'], entry['var'], ax_plot, color=line_colors[c])
        ax_plot.set_ylabel(entry['var'], fontsize=14)
        c += 1

    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc='upper left', fontsize=14)

    return fig