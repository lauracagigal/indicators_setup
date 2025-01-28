import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import plotly.graph_objects as go
from plotly.subplots import make_subplots  # Ensure this is imported
import plotly.colors

from scipy.stats import linregress

from .colors import get_df_col, plotting_style
from .core import fontsize


line_colors = get_df_col()
plotting_style()


def plot_timeseries_interactive(dict_plot, scatter_dict = None, trendline = False, ylims = None, figsize = (20, 5)):
    
    """
    Plots interactive time series data with optional scatter plots and trendlines.

    Parameters:
    - dict_plot (list): List of dictionaries containing information about the line plots.
        Each dictionary should have the following keys:
            - 'data' (pandas.DataFrame): Time series data.
            - 'var' (str): Variable to be plotted on the y-axis.
            - 'label' (str): Label for the legend.
            - 'ax' (int): Axis number (1 or 2) to plot the data on.
    - scatter_dict (list, optional): List of dictionaries containing information about the scatter plots.
        Each dictionary should have the following keys:
            - 'data' (pandas.DataFrame): Time series data.
            - 'var' (str): Variable to be plotted on the y-axis.
            - 'label' (str): Label for the legend.
    - trendline (bool, optional): Whether to plot trendlines for each line plot. Default is False.
    - ylims (tuple, optional): Tuple of y-axis limits for the line plots. Default is None.

    Returns:
    - fig (plotly.graph_objects.Figure): Interactive plotly figure object.
    """
    
    # Create a figure with two y-axes (secondary_y=True)
    fig = make_subplots(specs=[[{"secondary_y": True}]],)

    c = 0  # Color index to cycle through the color palette

    # Loop through each entry in the input list to plot data
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
            trendline_data = plot_trendline_interactive(entry['data'], entry['var'])  # Get trendline data
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
            # fig.update_yaxes(title_text=dict_plot[0]['var'], secondary_y=False)
            fig.update_yaxes(title_text=entry['var'], secondary_y=False)
        elif entry['ax'] == 2:
            fig.update_yaxes(title_text=entry['var'], secondary_y=True)
            # fig.update_yaxes(title_text=dict_plot[0]['var'], secondary_y=True)

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

    return fig


def plot_trendline_interactive(data, var):
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
    if p_value < 0.05:
        trendline_trace = go.Scatter(
        x=data.index,  # Trendline x values (original time index)
        y=trendline(time_num),  # Trendline y values based on the fit
        mode='lines',  # Display as a line
        line=dict(color='black'),
        name = 'Trendline - Significant (p < 0.05)'
    )
    else:
        trendline_trace = go.Scatter(
        x=data.index,  # Trendline x values (original time index)
        y=trendline(time_num),  # Trendline y values based on the fit
        mode='lines',  # Display as a line
        line=dict(color='black', dash='dot'),  # Black dotted line for the trendline
        name = 'Trendline - Not Significant (p > 0.05)'
        )
  

    return trendline_trace  # Return the trendline trace to be added to the plot


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