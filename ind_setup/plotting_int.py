import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import plotly.graph_objects as go
from plotly.subplots import make_subplots  # Ensure this is imported
import plotly.colors

from .colors import get_default_line_colors, plotting_style

line_colors = get_default_line_colors()
plotting_style()

def plot_timeseries_interactive(dict_plot, trendline = False, ylims = None):
    """
    Plots interactive line plots with optional trendlines using Plotly.

    Parameters:
    - dict_plot (list of dicts): A list where each dictionary contains the data for plotting.
        - 'data': The DataFrame containing the time series data to plot.
        - 'var': The column name of the variable to plot.
        - 'label': The label to display in the legend for each trace.
        - 'ax': The axis (1 for the primary axis, 2 for the secondary axis).
    """
    
    # Create a figure with two y-axes (secondary_y=True)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
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
                        line=dict(color=line_colors[c], dash='dash'),  # Trendline styling
                        name=f"{entry['label']} (Trend)",  # Label for trendline in the legend
                        showlegend=False  # Do not show the trendline in the legend
                    ),
                    secondary_y=secondary  # Attach to the secondary y-axis if needed
                )

        c += 1  # Increment color index for the next trace

        # Update y-axis labels
        if entry['ax'] == 1:
            fig.update_yaxes(title_text=dict_plot[0]['var'], secondary_y=False)
        elif entry['ax'] == 2:
            fig.update_yaxes(title_text=dict_plot[0]['var'], secondary_y=True)
        
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
        width=800,  # Set the width of the figure
        height=500,  # Set the height of the figure 
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

    # Create the trendline trace (no name assigned to avoid it showing in the legend)
    trendline_trace = go.Scatter(
        x=data.index,  # Trendline x values (original time index)
        y=trendline(time_num),  # Trendline y values based on the fit
        mode='lines',  # Display as a line
        line=dict(color='black', dash='dot'),  # Black dotted line for the trendline
    )

    return trendline_trace  # Return the trendline trace to be added to the plot