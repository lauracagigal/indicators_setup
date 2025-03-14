import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as sp
from scipy.stats import linregress

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from .colors import get_df_col, plotting_style
from .core import fontsize

line_colors = get_df_col()
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


    _, _, _, p_value, _ = linregress(time_num, data.values)
    if p_value < 0.05:
        ax.plot(data.index, trendline(time_num), color=color, linestyle='-', label='Trendline (p < 0.05 Significant)')
    else:
        ax.plot(data.index, trendline(time_num), color=color, linestyle=':', label='Trendline (p > 0.05 Not Significant)')
    
def plot_trendline_year(data, var, ax, color = 'k'):
    
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
    try:
        time_num = data.index.year
    except:
        time_num = time

    # Fit a trendline (degree 1 polynomial)
    coefficients = np.polyfit(time_num, data.values, 1)  # Linear fit

    trendline = np.poly1d(coefficients)  # Create trendline function

    change_rate = coefficients[0]

    _, _, _, p_value, _ = linregress(time_num, data.values)
    if p_value < 0.05:
        label = f'Trend (rate = {np.round(change_rate, 3)}/year) - Significant (p < 0.05)'
        ax.plot(time_num, trendline(time_num), color=color, linestyle='-', label=label)
    else:
        label = f'Trend (rate = {np.round(change_rate, 3)}/year) - Not Significant (p > 0.05)'
        ax.plot(time_num, trendline(time_num), color=color, linestyle=':', label=label)
    

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

def add_oni_cat(df1, lims = [-.5, .5]):
    """
    Adds a categorical column 'oni_cat' to the input DataFrame based on ONI values.
    
    Parameters:
    df1 (DataFrame): Input DataFrame containing the 'ONI' column.
    
    Returns:
    DataFrame: DataFrame with the additional 'oni_cat' column.
    """
    number_months  = 5
    df1['oni_cat'] = 0
    df1['oni_cat'] = np.where(df1.groupby(df1.index.year)['ONI'].transform(lambda x: x[x > lims[1]].count()) >= number_months, 1, df1['oni_cat'])
    df1['oni_cat'] = np.where(df1.groupby(df1.index.year)['ONI'].transform(lambda x: x[x < lims[0]].count()) >= number_months, -1, df1['oni_cat'])

    return df1

def plot_bar_probs(x, y, bar_label = None, labels = None, trendline = False, 
                   y_label = ' ', figsize = [7, 5], return_trend = False):
    """
    Plots a bar chart showing the distribution of wet days.

    Parameters:
    x (list): The x-axis values for the bar chart.
    y (list): The y-axis values for the bar chart.
    labels (list, optional): The labels for the x-axis ticks. Defaults to None.

    Returns:
    None
    """

    fig, ax = plt.subplots(figsize = figsize)
    if bar_label is not None:
        ax.bar(x = x, height = y, color=get_df_col()[0], edgecolor='white', alpha = .5, label = bar_label)
    else:   
        ax.bar(x = x, height = y, color=get_df_col()[0], edgecolor='white', alpha = .5)
    
    ax.set_ylabel(y_label, fontsize = fontsize)
    if labels:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize = fontsize)


    if trendline:
        time = x
        time_num = time
        # Fit a trendline (degree 1 polynomial)
        coefficients = np.polyfit(time_num, y, 1)  # Linear fit

        trendline = np.poly1d(coefficients)  # Create trendline function
        _, _, _, p_value, _ = linregress(time_num, y)


        change_rate = coefficients[0]
        trend = np.round(change_rate, 3)

        if p_value < 0.05:
            label = f'Trend (rate = {np.round(change_rate, 3)}/year) - Significant (p < 0.05)'
            ax.plot(time_num, trendline(time_num), color='k', linestyle='-', label=label)
        else:
            label = f'Trend (rate = {np.round(change_rate, 3)}/year) - Not Significant (p > 0.05)'
            ax.plot(time_num, trendline(time_num), color='k', linestyle=':', label= label)
        ax.legend(fontsize = fontsize)


    ax.grid(color = 'lightgrey', linestyle = ':', alpha = 0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
    
    return (fig, ax, trend) if return_trend else (fig, ax)





def plot_base_map(shp_eez = None, ax = None, figsize=[10, 6]):
    """
    Plots a map with optional EEZ shapefile overlay.

    Parameters:
    shp_eez (GeoDataFrame, optional): EEZ shapefile to overlay on the map. Default is None.
    ax (matplotlib.axes.Axes, optional): Axes object to plot on. If None, a new figure and axis will be created. Default is None.
    figsize (list, optional): Figure size in inches. Default is [10, 6].

    Returns:
    matplotlib.axes.Axes: The plotted axes object.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, subplot_kw={'projection': ccrs.PlateCarree()})

    # Set the extent of the map
    ax.coastlines(resolution='10m')
    ax.gridlines(draw_labels=True)

    if shp_eez is not None:
        shp_eez.boundary.plot(ax=ax, color='grey', linewidth=1.5)

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    return ax


def plot_map_subplots(data_an, var, shp_eez = None, cmap='RdBu_r', vmin=-.3, vmax=.3, 
                      sub_plot = None, figsize = (17, 20), cbar = None, cbar_pad = -0.02, titles = None):
    
    """
    Plots subplots of a map for each year in the given dataset.

    Parameters:
    - data_an (xarray.Dataset): The dataset containing the data to be plotted.

    Returns:
    None
    """

    
    lon_range = [data_an.longitude.min().values, data_an.longitude.max().values]
    lat_range = [data_an.latitude.min().values, data_an.latitude.max().values]

    a = list(data_an.dims.keys())
    a.remove('latitude')
    a.remove('longitude')
    dim_plot = a[0]

    if sub_plot is None:
        ns = int(np.ceil(np.sqrt(len(data_an[dim_plot]))))
        sub_plot = [ns, ns]
    
    fig, axs = plt.subplots(sub_plot[0], sub_plot[1], figsize=figsize, subplot_kw={'projection': ccrs.PlateCarree()}, )
    try:
        axs = axs.flatten()
    except:
        pass

    for i, y in enumerate(data_an[dim_plot]):
        # data_year = data_xr.isel(time=i)
        data_year = data_an.isel(**{dim_plot: i})
        ax = axs[i]
        if i < len(data_an[dim_plot]):
            ax = plot_base_map(shp_eez=shp_eez, ax = ax)
            im = ax.pcolor(data_year.longitude, data_year.latitude, data_year[var], transform=ccrs.PlateCarree(), 
                           cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_extent([lon_range[0], lon_range[1], lat_range[0], lat_range[1]], crs=ccrs.PlateCarree())

            if titles is not None:
                ax.set_title(titles[i], fontsize=fontsize)
            else:
                try:
                    ax.set_title(f'Year {y.dt.year.values}', fontsize=fontsize)
                except:
                    try:
                        ax.set_title(data_an.labels.values[i], fontsize=fontsize)
                    except:
                        ax.set_title(f'{dim_plot}: {data_year[dim_plot].values}', fontsize=fontsize)
            ax.spines['geo'].set_visible(False)

    if sub_plot[0]*sub_plot[1] > len(data_an[dim_plot]):
            for j in range(len(data_an[dim_plot]), len(axs)):
                fig.delaxes(axs[j]) 

    # plt.colorbar(im, ax=axs, label='Phytoplankton (µm)', orientation='horizontal')
    plt.tight_layout(h_pad=0.2, w_pad=0.2)
    plt.tight_layout()

    if cbar is not None:
        if var == 'o2':
            label = 'Oxygen (µmol/kg)'
        elif var == 'MD50':
            label = 'MD50 (µm)'
        else:
            label = var

        fig.colorbar(im, ax=axs, label=label, orientation='horizontal', shrink=0.6, 
                     pad = cbar_pad).set_label(label, fontsize = fontsize)
    plt.show()

def plot_oni_index(df1, lims = [-.5, .5]):

    fig, ax = plt.subplots(figsize = [15, 6])
    df1.plot(ax = ax, color = get_df_col()[0], lw = 2)
    ax.hlines([lims[0], 0, lims[1]], df1.index[0], df1.index[-1], 
            color = ['grey', get_df_col()[1], 'grey', get_df_col()[1]], 
            linestyle = '--', label = 'Thresholds')

    ax.fill_between(df1.index, lims[1], df1.ONI, where = (df1.ONI > lims[1]), color = 'lightcoral', alpha = 0.7)
    ax.fill_between(df1.index, df1.ONI, lims[0], where = (df1.ONI < lims[0]), color = 'lightblue', alpha = 0.7)

    ax.legend(fontsize = fontsize, ncol = 2)
    ax.set_ylabel('ONI Index', fontsize = fontsize)


def plot_bar_probs_ONI(df2, var, y_label = ''):
    """
    Plots a bar chart of the mean annual precipitation with respect to the ONI categories.

    Parameters:
    df2 (pandas.DataFrame): The DataFrame containing the data.
    var (str): The variable to be plotted.

    Returns:
    None
    """

    try:
        x = df2.index.year
    except:
        x = df2.index
    y = df2[var]

    # Map ONI categories to colors
    categories = [-1, 0, 1]
    colors = ['lightblue', 'lightgray', 'lightcoral']
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm([category - 0.5 for category in categories] + [1.5], cmap.N)  # Shift for centered labels

    # Get colors for bars
    colors_bars = [cmap(norm(value)) for value in df2['oni_cat']]

    fig, ax = plt.subplots(figsize=(15, 6))

    # Plot bars
    bars = ax.bar(x=x, height=y, color=colors_bars, edgecolor='white', alpha=0.7)

    ax.set_ylabel(y_label, fontsize=fontsize)
    ax.set_xlabel('Year', fontsize=fontsize)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # No data needed, just a mapping
    cbar = fig.colorbar(sm, ax=ax, orientation='vertical')

    # Set custom tick positions and labels
    tick_positions = categories  # Use category values as tick positions
    cbar.set_ticks(tick_positions)
    # cbar.set_ticklabels([f'ONI < {categories[0]}', 'Neutral', f'ONI > {categories[2]}'], fontsize=fontsize)
    # cbar.set_ticklabels([f'ONI < -0.5 ', 'Neutral', f'ONI > 0.5'], fontsize=fontsize)
    cbar.set_ticklabels(['La Niña', 'Neutral', 'El Niño'], fontsize=fontsize)
    # cbar.set_label('ONI based categories', fontsize=fontsize)

    # Format plot
    ax.grid(color='lightgrey', linestyle=':', alpha=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # Add trendline
    plot_trendline_year(df2, var, ax, color='k')
    ax.legend(fontsize = fontsize)
    plt.tight_layout()  # Adjust layout to fit everything
    # plt.show()

    return fig


def plot_tc_categories_trend(tcs_sel_params):

    fig, ax = plt.subplots(1, figsize=(15, 4))
    df_tcs = tcs_sel_params.to_dataframe()
    df_tcs['year'] = df_tcs.dmin_date.dt.year
    df_tcs.groupby('year').category.value_counts().unstack().plot(ax = ax, kind = 'bar', stacked = True, color = ['lightgrey', 'green', 'yellow', 'orange', 'red', 'purple', 'black'])
    ax.set_ylabel('Counts', fontsize = 14)
    ax.set_xlabel('Year', fontsize = 14)
    ax.legend(ncols = 6).set_title('Category', prop = {'size': 12})

    ax.grid(':', color = 'lightgrey', alpha = 0.5)
    #trendline
    x = df_tcs.groupby('year').count().index  # Esto es lo que Pandas usa en el bar plot
    x = np.arange(len(x))
    y = df_tcs.groupby('year').month.count().values


    coefficients = np.polyfit(x, y, 1)  # Linear fit
    trendline = np.poly1d(coefficients)  # Create trendline function

    _, _, _, p_value, _ = linregress(x, y)

    change_rate = coefficients[0]
    trend = np.round(change_rate, 3)


    if p_value < 0.05:
        label = f'Trend (rate = {np.round(change_rate, 2)}/year) - Significant (p < 0.05)'
        ax.plot(x, trendline(x), color='k', linestyle='-', label=label)
    else:
        label = f'Trend (rate = {np.round(change_rate, 2)}/year) - Not Significant (p > 0.05)'
        ax.plot(x, trendline(x), color='k', linestyle=':', label= label)

    ax.legend(fontsize = 12, ncol = 7)