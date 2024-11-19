import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex

import seaborn as sns
import scipy.stats as sp

import plotly.graph_objects as go
from plotly.subplots import make_subplots  # Ensure this is imported
import plotly.colors



def get_default_line_colors():
    """
    Returns the default line colors for plotting.

    Returns:
        list: A list of default line colors.
    """
    colors = plotly.colors.qualitative.Plotly

    palette = sns.color_palette("gist_ncar", n_colors=100)
    palette = [to_hex(color) for color in palette]
    
    colors.extend(palette)

    return colors


def plotting_style():
    """
    Sets the default plotting style using Seaborn.

    Returns:
        None
    """
    sns.set_style("whitegrid")