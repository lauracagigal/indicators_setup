import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import plotly.graph_objects as go
from plotly.subplots import make_subplots  # Ensure this is imported
import plotly.colors

from scipy.stats import linregress

import plotly.io as pio
from PIL import Image
import io

from .colors import get_df_col, plotting_style
from .core import fontsize


line_colors = get_df_col()
plotting_style()


def plot_timeseries_interactive(dict_plot, scatter_dict=None, trendline=False, ylims=None, figsize=(20, 5),
                               return_trend=False, label_yaxes=None):
    """
    Plots interactive timeseries data with optional scatter plots and trendlines.

    Parameters:
    - dict_plot (list): List of dictionaries containing the data and settings for each line plot.
    - scatter_dict (list, optional): List of dictionaries containing the data and settings for each scatter plot. Default is None.
    - trendline (bool, optional): Flag indicating whether to include trendlines. Default is False.
    - ylims (tuple, optional): Tuple specifying the y-axis limits. Default is None.
    - figsize (tuple, optional): Tuple specifying the figure size. Default is (20, 5).
    - return_trend (bool, optional): Flag indicating whether to return the trendline data. Default is False.
    - label_yaxes (str, optional): Label for the y-axis. Default is None.

    Returns:
    - fig (plotly.graph_objects.Figure): The plotly figure object.
    - TRENDS (list): List of trendline data if return_trend is True, otherwise an empty list.
    """
    
    # Create a figure with two y-axes (secondary_y=True)
    fig = make_subplots(specs=[[{"secondary_y": True}]],)

    c = 0  # Color index to cycle through the color palette

    # Loop through each entry in the input list to plot data
    TRENDS = []
    for entry in dict_plot:  
        # Determine if the trace will be plotted on the secondary y-axis
        secondary = entry['ax'] == 2
        
        # Add the line plot trace to the figure
        fig.add_trace(
            go.Scatter(
                x=entry['data'].index,  # X-axis data (time index)
                y=entry['data'][entry['var']],  # Y-axis data (selected variable)
                name=entry['label'],  # Label for the legend
                line=dict(color=line_colors[c])  # Line color based on the color index
            ),
            secondary_y=secondary  # Attach to the secondary y-axis if needed
        )

        if trendline:
            # Add a trendline if applicable
            try:
                tr = entry['trendline']
            except:
                tr = True
            if tr:
                if return_trend:
                    trendline_data, trend = plot_trendline_interactive(entry['data'], entry['var'], return_trend = return_trend)  # Get trendline data
                    TRENDS.append(trend)
                else:
                    trendline_data = plot_trendline_interactive(entry['data'], entry['var'], return_trend = return_trend)
                if trendline_data:  # If a trendline is returned, add it to the plot
                    fig.add_trace(
                        go.Scatter(
                            x=trendline_data['x'],  # Trendline x values
                            y=trendline_data['y'],  # Trendline y values
                            mode='lines',  # Display as a line
                            line=trendline_data['line'],  # Trendline styling
                            name=trendline_data['name'],  # Label for trendline in the legend
                            showlegend=True  # Do not show the trendline in the legend
                        ),
                        secondary_y=secondary  # Attach to the secondary y-axis if needed
                    )

        c += 1  # Increment color index for the next trace

        # Update y-axis labels
        if entry['ax'] == 1:
            yaxis_title = label_yaxes if label_yaxes is not None else entry['var']
            fig.update_yaxes(title_text=yaxis_title, secondary_y=False)
        elif entry['ax'] == 2:
            fig.update_yaxes(title_text=entry['var'], secondary_y=True)

    if scatter_dict:

        for entry in scatter_dict:

            fig.add_trace(
                go.Scatter(
                    x=entry['data'].index,  # X-axis data (time index)
                    y=entry['data'][entry['var']],  # Y-axis data (selected variable)
                    mode='markers', 
                    marker=dict(
                        size=10,  # Tamaño de los marcadores
                        color=entry['data'][entry['var']],  # Colorear los marcadores en función de otra variable
                        colorscale='rainbow',  # Escala de colores (puedes elegir otra)
                        showscale=True  # Mostrar la barra de colores
                    
                    ),
                    name= entry['label'],  # Label for the legend
                ),
            )
        
    # Update general layout of the plot
    fig.update_layout(
        legend=dict(
        orientation='h',  # Set legend orientation to horizontal
        yanchor='bottom',  # Anchor legend at the bottom of its position
        y=1.1,  # Adjust vertical position (1.1 moves it above the plot)
        xanchor='center',  # Center the legend horizontally
        x=0.5,  # Place the legend in the center of the plot width
        font=dict(size=12)  # Set font size for legend labels
        ),
        width=figsize[0]*40,  # Set the width of the figure
        height=figsize[1]*40,  # Set the height of the figure 
        title=" ",  # No title for now, can be updated as needed
        xaxis_title="Time",  # X-axis label
        template="plotly_white",  # Use a clean, white background template
    )

    # Update the y-axis limits if ylims is provided
    if ylims is not None:
        if entry['ax'] == 1:
            fig.update_yaxes(range=ylims, secondary_y=False)
        elif entry['ax'] == 2:
            fig.update_yaxes(range=ylims, secondary_y=True)

    # Display the plot
    fig.show()
    # plt.show()

    return (fig, TRENDS) if return_trend else fig




def plot_trendline_interactive(data, var, return_trend = False):
    """
    Fits and returns a linear trendline for the specified variable in the data.

    Parameters:
    - data (pandas.DataFrame): The DataFrame containing the variable to fit a trendline to.
    - var (str): The column name of the variable to fit a trendline for.

    Returns:
    - trendline_trace (go.Scatter): A Plotly trace representing the trendline.
    """

    # Drop missing values for the specified variable
    data = data[var].dropna()

    # Convert the time index to numerical values for fitting
    time = data.index.values
    time_num = time.view('int64') // 1e9  # Convert to seconds since epoch (numeric format)

    # Fit a linear trendline (degree 1 polynomial)
    coefficients = np.polyfit(time_num, data.values, 1)  # Linear fit
    trendline = np.poly1d(coefficients)  # Create a polynomial function for the trendline

    _, _, _, p_value, _ = linregress(time_num, data.values)

    #Since time is in seconds:
    change_rate = coefficients[0] * 3600 * 24 * 365

    if p_value < 0.05:
        trendline_trace = go.Scatter(
        x=data.index,  # Trendline x values (original time index)
        y=trendline(time_num),  # Trendline y values based on the fit
        mode='lines',  # Display as a line
        line=dict(color='black'),
        name = f'Trend (rate = {np.round(change_rate, 3)}/year) - Significant (p < 0.05)'
    )
    else:
        trendline_trace = go.Scatter(
        x=data.index,  # Trendline x values (original time index)
        y=trendline(time_num),  # Trendline y values based on the fit
        mode='lines',  # Display as a line
        line=dict(color='black', dash='dot'),  # Black dotted line for the trendline
        name = f'Trend (rate = {np.round(change_rate, 3)}/year) - Not Significant (p > 0.05)'
        )

    return (trendline_trace, np.round(change_rate, 3)) if return_trend else trendline_trace

def plot_oni_index_th(df1, lims = [-.5, .5]):
    """
    Plots the ONI (Oceanic Niño Index) index with threshold lines and fill areas.

    Parameters:
    - df1 (DataFrame): The DataFrame containing the ONI index data.
    - lims (list, optional): The threshold values for the fill areas. Default is [-0.5, 0.5].

    Returns:
    None
    """

    # Create the base figure
    fig = go.Figure()

    # Add the main line plot
    fig.add_trace(go.Scatter(
        x=df1.index,
        y=df1['ONI'],
        mode='lines',
        line=dict(color=get_df_col()[0], width=2),
        name='ONI'
    ))

    # Add horizontal lines (Thresholds)
    hline_colors = ['grey', get_df_col()[1], 'grey']
    hline_labels = ['Threshold', 'Zero Line', 'Threshold']
    for i, y in enumerate([lims[0], 0, lims[1]]):
        fig.add_trace(go.Scatter(
            x=[df1.index[0], df1.index[-1]],
            y=[y, y],
            mode='lines',
            line=dict(color=hline_colors[i], dash='dash'),
            name=hline_labels[i],
            showlegend=(i == 0)  # Only show legend once
        ))

    # Add fill for values greater than lims[1]
    fig.add_trace(go.Scatter(
        x=list(df1.index) + list(df1.index[::-1]),
        y=list(df1['ONI'].where(df1['ONI'] > lims[1], lims[1])) + [lims[1]] * len(df1),
        fill='toself',
        mode='none',
        fillcolor='lightcoral',
        opacity=0.7,
        name='Above Threshold'
    ))

    # Add fill for values less than lims[0]
    fig.add_trace(go.Scatter(
        x=list(df1.index) + list(df1.index[::-1]),
        y=[lims[0]] * len(df1) + list(df1['ONI'].where(df1['ONI'] < lims[0], lims[0]))[::-1],
        fill='toself',
        mode='none',
        fillcolor='lightblue',
        opacity=0.7,
        name='Below Threshold'
    ))

    # Update layout
    fig.update_layout(
        title='ONI Index Plot',
        xaxis_title='Time',
        yaxis_title='ONI Index',
        font=dict(size=fontsize),
        legend=dict(orientation='h', y=-0.2),
        plot_bgcolor='white',
        margin=dict(l=50, r=50, t=50, b=50),
    )

    # Show the plot
    fig.show()


def fig_int_to_glue(fig):
    """
    Converts an interactive plotly  figure to a PIL image and returns it.

    Parameters:
    fig (matplotlib.figure.Figure): The interactive plotly figure to convert.

    Returns:
    PIL.Image.Image: The converted PIL image.
    """
    # Guardar la figura en un buffer en memoria
    img_bytes = io.BytesIO()
    pio.write_image(fig, img_bytes, format="png")

    # Abrir la imagen con PIL y pasarla a glue
    img = Image.open(img_bytes)

    return img