import df2img

def plot_df_table(df, header_fill = "#d4b9d9", header_font ="#3c453a",
                  cell_fill = ["#ffe4dc", "#ddeed9"], cell_font = "#3c453a",
                  figsize = (800, 300)):
    """
    Plots a dataframe as a stylized table.

    Parameters:
    - df: pandas DataFrame
        The dataframe to be plotted.
    - header_fill: str, optional
        The fill color for the table header. Default is "#d4b9d9".
    - header_font: str, optional
        The font color for the table header. Default is "#3c453a".
    - cell_fill: list of str, optional
        The fill colors for the table cells. Default is ["#ffe4dc", "#ddeed9"].
    - cell_font: str, optional
        The font color for the table cells. Default is "#3c453a".
    - figsize: tuple, optional
        The size of the resulting figure. Default is (800, 300).

    Returns:
    - fig: matplotlib.figure.Figure
        The resulting figure object.
    """
    
    fig = df2img.plot_dataframe(
    df,
    tbl_header={
        "height": 20,
        "line_width": 3,
        "align": "center",
        'fill_color': header_fill,
        "font_color": header_font,
    },
    tbl_cells={
        "align": "center",
        "fill_color": cell_fill,
        "font_color": cell_font,
        "height": 30,
        "line_width": 3,
    }, 
    fig_size=figsize,
    plotly_renderer='png',)


    return fig