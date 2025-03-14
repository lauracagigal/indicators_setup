import df2img
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_data_metrics(st_data, var):
    """
    Calculate various metrics for a given variable in a dataset.

    Parameters:
    st_data (pd.DataFrame): The dataset containing the variable.
    var (str): The name of the variable to calculate metrics for.

    Returns:
    pd.DataFrame: A DataFrame containing the calculated metrics for the variable.
    """
    
    mean = np.nanmean(st_data[var])
    std = np.nanstd(st_data[var])
    max_val = np.nanmax(st_data[var])
    min_val = np.nanmin(st_data[var])
    median = np.nanmedian(st_data[var])
    range_val = max_val - min_val
    trend = np.polyfit(st_data.index.year, st_data[var], 1)[0]
    
    # Extreme events
    threshold = mean + 2 * std
    extreme_days = len(st_data[var] > threshold)
    
    # Percentiles
    p10 = np.nanpercentile(st_data[var], 10)
    p90 = np.nanpercentile(st_data[var], 90)
    
    
    # Compile metrics
    df = pd.DataFrame({
        'Mean': mean,
        'Median': median,
        'Standard deviation': std,
        'Maximum': max_val,
        'Minimum': min_val,
        'Range': range_val,
        '10th Percentile': p10,
        '90th Percentile': p90,
        'Trend': trend,
        'Extreme Days (>2σ)': extreme_days,
    }, index=[var])
    
    return np.round(df, 3)


def plot_df_table(df, header_fill = "lavender", header_font ="#3c453a",
                #   cell_fill = ["#ffe4dc", "#ddeed9"], cell_font = "#3c453a",
                  cell_fill = ["whitesmoke", "ghostwhite"], cell_font = "#3c453a",
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


def table_temperature_summary(st_data):
    """
    Calculate various metrics for a given variable in a dataset.

    Parameters:
    st_data (pd.DataFrame): The dataset containing the variable.
    var (str): The name of the variable to calculate metrics for.

    Returns:
    pd.DataFrame: A DataFrame containing the calculated metrics for the variable.
    """

    annual_mean = np.nanmean(st_data['TMEAN'].resample('Y').mean())
    annual_min = np.nanmin(st_data['TMEAN'].resample('Y').mean())
    annual_max = np.nanmax(st_data['TMEAN'].resample('Y').mean())

    annual_max_date = st_data['TMAX'].resample('Y').mean().idxmax().year
    annual_max_value = st_data['TMAX'].resample('Y').mean().max()
    annual_min_date = st_data['TMIN'].resample('Y').mean().idxmin().year
    annual_min_value = st_data['TMIN'].resample('Y').mean().min()
    
    
    # Compile metrics
    df = pd.DataFrame({
        'Annual Mean Temperature': annual_mean,
        'Range of Mean Annual Temperature': str([f"{float(annual_min):.2f}", f"{float(annual_max):.2f}"]),
        '_ _': '_ _',
        'Minimum Temperature Year': annual_min_date,
        'Annual Minimum Temperature': annual_min_value,
        '__': '__',
        'Maximum Temperature Year': annual_max_date,
        'Annual maximum temperature': annual_max_value,
       
    }, index=['Stats'])
    df.index.name = ' '
    
    return np.round(df, 3)


def table_temperature_summary(st_data):
    """
    Calculate various metrics for a given variable in a dataset.

    Parameters:
    st_data (pd.DataFrame): The dataset containing the variable.
    var (str): The name of the variable to calculate metrics for.

    Returns:
    pd.DataFrame: A DataFrame containing the calculated metrics for the variable.
    """

    annual_mean = np.nanmean(st_data['TMEAN'].resample('Y').mean())
    annual_min = np.nanmin(st_data['TMEAN'].resample('Y').mean())
    annual_max = np.nanmax(st_data['TMEAN'].resample('Y').mean())

    annual_max_date = st_data['TMAX'].resample('Y').mean().idxmax().year
    annual_max_value = st_data['TMAX'].resample('Y').mean().max()
    annual_min_date = st_data['TMIN'].resample('Y').mean().idxmin().year
    annual_min_value = st_data['TMIN'].resample('Y').mean().min()
    
    
    # Compile metrics
    df = pd.DataFrame({

        'Warmest day on record': st_data['TMAX'].idxmax().strftime('%Y-%m-%d'),
        'Coldest day on record': st_data['TMIN'].idxmin().strftime('%Y-%m-%d'),

        'Annual Mean Temperature': annual_mean,
        'Range of Mean Annual Temperature': str([f"{float(annual_min):.2f}", f"{float(annual_max):.2f}"]),
        '_ _': '_ _',
        'Minimum Temperature Year': annual_min_date,
        'Annual Minimum Temperature': annual_min_value,
        '__': '__',
        'Maximum Temperature Year': annual_max_date,
        'Annual maximum temperature': annual_max_value,
       
    }, index=['Stats'])
    df.index.name = ' '
    
    return np.round(df, 3)



def table_hot_cold_summary(annual_hot, annual_cold, TRENDS):
    """
    Calculate various metrics for a given variable in a dataset.

    Parameters:
    st_data (pd.DataFrame): The dataset containing the variable.
    var (str): The name of the variable to calculate metrics for.

    Returns:
    pd.DataFrame: A DataFrame containing the calculated metrics for the variable.
    """
    
    # Compile metrics
    df = pd.DataFrame({
        'Average number of hot days': np.nanmean(annual_hot),
        'Change in average number of hot days per decade': float(TRENDS[1]*10),
        'Average number of hot days (1951-1962)': np.nanmean(annual_hot.loc['1951':'1962']),
        'Average number of hot days (1981-1992)': np.nanmean(annual_hot.loc['1981':'1992']),
        'Average number of hot days (2012 - 2021)': np.nanmean(annual_hot.loc['2012':'2021']),
        'Maximum number of hot days': int(np.nanmax(annual_hot)),
        'Year with maximum number of hot days': annual_hot['Perc_Anom'].idxmax().year,
        '-- ': '--', 

        'Average number of cold nights': np.nanmean(annual_cold),
        'Change in average number of cold nights per decade': float(TRENDS[0]*10),
        'Average number of cold nights (1951-1962)': np.nanmean(annual_cold.loc['1951':'1962']),
        'Average number of cold nights (1981-1992)': np.nanmean(annual_cold.loc['1981':'1992']),
        'Average number of cold nights (2012 - 2021)': np.nanmean(annual_cold.loc['2012':'2021']),
        'Maximum number of cold nights': int(np.nanmax(annual_cold)),
        'Year with maximum number of cold nights': annual_cold['Perc_Anom'].idxmax().year,
        

    }, index=['Stats'])
    df.index.name = ' '
    
    return np.round(df, 3)



def table_rain_a_summary(data):
    """
    Generate a summary table for rainfall data.

    Parameters:
    data (pd.DataFrame): DataFrame containing rainfall data.

    Returns:
    pd.DataFrame: Summary table with maximum accumulated rainfall and maximum daily precipitation.
    """

    datag = (data.groupby(data.index.year).sum()/ data.groupby(data.index.year).count()) * 365
    datag.index = pd.to_datetime(datag.index, format = '%Y')

    perc_wet_dry = data.groupby('wet_day').count()['PRCP'].values/len(data) * 100

    daily_max = data.PRCP.max()
    daily_max_date = data.PRCP.idxmax().date()

    data_2 = data.loc[data['wet_day_t'] == 1][['PRCP']]
    data_over_th = data_2.groupby(data_2.index.year).count()
    data_over_th.index = pd.to_datetime(data_over_th.index, format = '%Y')
    year_th = data_over_th['PRCP'].idxmax().year

    acum_max = datag.PRCP.max()
    acum__max_date = datag.PRCP.idxmax().year

    # Compile metrics
    df = pd.DataFrame({
        '% dry days [<1mm]': perc_wet_dry[0],
        '% wet days [>1mm]': perc_wet_dry[1],
        '-':'--',
        'Maximum Accumulated Rainfall [mm]': acum_max,
        'Maximum Accumulated Rainfall year': acum__max_date,
        '--':'--',
        'Year with more days over 95th percentile': year_th,
        '---':'--',
        'Maximum Daily Precipitation [mm]': daily_max,
        'Maximum Daily Precipitation Date': daily_max_date,
        
    }, index=['Stats'])
    df.index.name = ' '

    return np.round(df, 2)


def table_rain_dry_summary(data):
    """
    Generate a summary table for rainfall data.

    Parameters:
    data (pd.DataFrame): DataFrame containing rainfall data.

    Returns:
    pd.DataFrame: Summary table with maximum accumulated rainfall and maximum daily precipitation.
    """

    data_dry = data.groupby(data.index.year)[['consecutive_days']].max()


    number_max = data_dry.consecutive_days.max()
    number_max_date = data_dry.consecutive_days.idxmax()

    number_mean = data_dry.consecutive_days.mean()
 

    # Compile metrics
    df = pd.DataFrame({

        'Maximum Number of Consecutive Dry Days': number_max,
        'Year': number_max_date,

        'Mean Number of Consecutive Dry Days': number_mean,
        
    }, index=['Stats'])
    df.index.name = ' '

    return np.round(df, 2)