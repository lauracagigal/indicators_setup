import df2img
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# def get_data_metrics(st_data, var):
#     """
#     Calculate various metrics for a given variable in a dataset.

#     Parameters:
#     st_data (pd.DataFrame): The dataset containing the variable.
#     var (str): The name of the variable to calculate metrics for.

#     Returns:
#     pd.DataFrame: A DataFrame containing the calculated metrics for the variable.
#     """
    
#     mean = np.nanmean(st_data[var])
#     std = np.nanstd(st_data[var])
#     max_val = np.nanmax(st_data[var])
#     min_val = np.nanmin(st_data[var])
#     median = np.nanmedian(st_data[var])
#     range_val = max_val - min_val
#     trend = np.polyfit(st_data.index.year, st_data[var], 1)[0]
    
#     # Extreme events
#     threshold = mean + 2 * std
#     extreme_days = len(st_data[var] > threshold)
    
#     # Percentiles
#     p10 = np.nanpercentile(st_data[var], 10)
#     p90 = np.nanpercentile(st_data[var], 90)
    
    
#     # Compile metrics
#     df = pd.DataFrame({
#         'Mean': mean,
#         'Median': median,
#         'Standard deviation': std,
#         'Maximum': max_val,
#         'Minimum': min_val,
#         'Range': range_val,
#         '10th Percentile': p10,
#         '90th Percentile': p90,
#         'Trend': trend,
#         'Extreme Days (>2σ)': extreme_days,
#     }, index=[var])
    
#     return np.round(df, 3)


# def plot_df_table(df, header_fill = "lavender", header_font ="#3c453a",
#                 #   cell_fill = ["#ffe4dc", "#ddeed9"], cell_font = "#3c453a",
#                   cell_fill = ["whitesmoke", "ghostwhite"], cell_font = "#3c453a",
#                   figsize = (800, 300)):
#     """
#     Plots a dataframe as a stylized table.

#     Parameters:
#     - df: pandas DataFrame
#         The dataframe to be plotted.
#     - header_fill: str, optional
#         The fill color for the table header. Default is "#d4b9d9".
#     - header_font: str, optional
#         The font color for the table header. Default is "#3c453a".
#     - cell_fill: list of str, optional
#         The fill colors for the table cells. Default is ["#ffe4dc", "#ddeed9"].
#     - cell_font: str, optional
#         The font color for the table cells. Default is "#3c453a".
#     - figsize: tuple, optional
#         The size of the resulting figure. Default is (800, 300).

#     Returns:
#     - fig: matplotlib.figure.Figure
#         The resulting figure object.
#     """
    
#     fig = df2img.plot_dataframe(
#     df,
#     tbl_header={
#         "height": 20,
#         "line_width": 3,
#         "align": "center",
#         'fill_color': header_fill,
#         "font_color": header_font,
#     },
#     tbl_cells={
#         "align": "center",
#         "fill_color": cell_fill,
#         "font_color": cell_font,
#         "height": 30,
#         "line_width": 3,
#     }, 
#     fig_size=figsize,
#     plotly_renderer='png',)


#     return fig


# def table_temperature_summary(st_data):
#     """
#     Calculate various metrics for a given variable in a dataset.

#     Parameters:
#     st_data (pd.DataFrame): The dataset containing the variable.
#     var (str): The name of the variable to calculate metrics for.

#     Returns:
#     pd.DataFrame: A DataFrame containing the calculated metrics for the variable.
#     """

#     annual_mean = np.nanmean(st_data['TMEAN'].resample('Y').mean())
#     annual_min = np.nanmin(st_data['TMEAN'].resample('Y').mean())
#     annual_max = np.nanmax(st_data['TMEAN'].resample('Y').mean())

#     annual_max_date = st_data['TMAX'].resample('Y').mean().idxmax().year
#     annual_max_value = st_data['TMAX'].resample('Y').mean().max()
#     annual_min_date = st_data['TMIN'].resample('Y').mean().idxmin().year
#     annual_min_value = st_data['TMIN'].resample('Y').mean().min()
    
    
#     # Compile metrics
#     df = pd.DataFrame({
#         'Annual Mean Temperature': annual_mean,
#         'Range of Mean Annual Temperature': str([f"{float(annual_min):.2f}", f"{float(annual_max):.2f}"]),
#         '_ _': '_ _',
#         'Minimum Temperature Year': annual_min_date,
#         'Annual Minimum Temperature': annual_min_value,
#         '__': '__',
#         'Maximum Temperature Year': annual_max_date,
#         'Annual maximum temperature': annual_max_value,
       
#     }, index=['Stats'])
#     df.index.name = ' '
    
#     return np.round(df, 3)


# def table_temperature_summary(st_data):
#     """
#     Calculate various metrics for a given variable in a dataset.

#     Parameters:
#     st_data (pd.DataFrame): The dataset containing the variable.
#     var (str): The name of the variable to calculate metrics for.

#     Returns:
#     pd.DataFrame: A DataFrame containing the calculated metrics for the variable.
#     """

#     annual_mean = np.nanmean(st_data['TMEAN'].resample('Y').mean())
#     annual_min = np.nanmin(st_data['TMEAN'].resample('Y').mean())
#     annual_max = np.nanmax(st_data['TMEAN'].resample('Y').mean())

#     annual_max_date = st_data['TMAX'].resample('Y').mean().idxmax().year
#     annual_max_value = st_data['TMAX'].resample('Y').mean().max()
#     annual_min_date = st_data['TMIN'].resample('Y').mean().idxmin().year
#     annual_min_value = st_data['TMIN'].resample('Y').mean().min()
    
    
#     # Compile metrics
#     df = pd.DataFrame({

#         'Warmest day on record': st_data['TMAX'].idxmax().strftime('%Y-%m-%d'),
#         'Coldest day on record': st_data['TMIN'].idxmin().strftime('%Y-%m-%d'),

#         'Annual Mean Temperature': annual_mean,
#         'Range of Mean Annual Temperature': str([f"{float(annual_min):.2f}", f"{float(annual_max):.2f}"]),
#         '_ _': '_ _',
#         'Minimum Temperature Year': annual_min_date,
#         'Annual Minimum Temperature': annual_min_value,
#         '__': '__',
#         'Maximum Temperature Year': annual_max_date,
#         'Annual maximum temperature': annual_max_value,
       
#     }, index=['Stats'])
#     df.index.name = ' '
    
#     return np.round(df, 3)



# def table_hot_cold_summary(annual_hot, annual_cold, TRENDS):
#     """
#     Calculate various metrics for a given variable in a dataset.

#     Parameters:
#     st_data (pd.DataFrame): The dataset containing the variable.
#     var (str): The name of the variable to calculate metrics for.

#     Returns:
#     pd.DataFrame: A DataFrame containing the calculated metrics for the variable.
#     """
    
#     # Compile metrics
#     df = pd.DataFrame({
#         'Average number of hot days': np.nanmean(annual_hot),
#         'Change in average number of hot days per decade': float(TRENDS[1]*10),
#         'Average number of hot days (1951-1962)': np.nanmean(annual_hot.loc['1951':'1962']),
#         'Average number of hot days (1981-1992)': np.nanmean(annual_hot.loc['1981':'1992']),
#         'Average number of hot days (2012 - 2021)': np.nanmean(annual_hot.loc['2012':'2021']),
#         'Maximum number of hot days': int(np.nanmax(annual_hot)),
#         'Year with maximum number of hot days': annual_hot['Perc_Anom'].idxmax().year,
#         '-- ': '--', 

#         'Average number of cold nights': np.nanmean(annual_cold),
#         'Change in average number of cold nights per decade': float(TRENDS[0]*10),
#         'Average number of cold nights (1951-1962)': np.nanmean(annual_cold.loc['1951':'1962']),
#         'Average number of cold nights (1981-1992)': np.nanmean(annual_cold.loc['1981':'1992']),
#         'Average number of cold nights (2012 - 2021)': np.nanmean(annual_cold.loc['2012':'2021']),
#         'Maximum number of cold nights': int(np.nanmax(annual_cold)),
#         'Year with maximum number of cold nights': annual_cold['Perc_Anom'].idxmax().year,
        

#     }, index=['Stats'])
#     df.index.name = ' '
    
#     return np.round(df, 3)



# def table_rain_a_summary(data):
#     """
#     Generate a summary table for rainfall data.

#     Parameters:
#     data (pd.DataFrame): DataFrame containing rainfall data.

#     Returns:
#     pd.DataFrame: Summary table with maximum accumulated rainfall and maximum daily precipitation.
#     """

#     datag = (data.groupby(data.index.year).sum()/ data.groupby(data.index.year).count()) * 365
#     datag.index = pd.to_datetime(datag.index, format = '%Y')

#     perc_wet_dry = data.groupby('wet_day').count()['PRCP'].values/len(data) * 100

#     daily_max = data.PRCP.max()
#     daily_max_date = data.PRCP.idxmax().date()

#     data_2 = data.loc[data['wet_day_t'] == 1][['PRCP']]
#     data_over_th = data_2.groupby(data_2.index.year).count()
#     data_over_th.index = pd.to_datetime(data_over_th.index, format = '%Y')
#     year_th = data_over_th['PRCP'].idxmax().year

#     acum_max = datag.PRCP.max()
#     acum__max_date = datag.PRCP.idxmax().year

#     # Compile metrics
#     df = pd.DataFrame({
#         '% dry days [<1mm]': perc_wet_dry[0],
#         '% wet days [>1mm]': perc_wet_dry[1],
#         '-':'--',
#         'Maximum Accumulated Rainfall [mm]': acum_max,
#         'Maximum Accumulated Rainfall year': acum__max_date,
#         '--':'--',
#         'Year with more days over 95th percentile': year_th,
#         '---':'--',
#         'Maximum Daily Precipitation [mm]': daily_max,
#         'Maximum Daily Precipitation Date': daily_max_date,
        
#     }, index=['Stats'])
#     df.index.name = ' '

#     return np.round(df, 2)


# def table_rain_dry_summary(data):
#     """
#     Generate a summary table for rainfall data.

#     Parameters:
#     data (pd.DataFrame): DataFrame containing rainfall data.

#     Returns:
#     pd.DataFrame: Summary table with maximum accumulated rainfall and maximum daily precipitation.
#     """

#     data_dry = data.groupby(data.index.year)[['consecutive_days']].max()


#     number_max = data_dry.consecutive_days.max()
#     number_max_date = data_dry.consecutive_days.idxmax()

#     number_mean = data_dry.consecutive_days.mean()
 

#     # Compile metrics
#     df = pd.DataFrame({

#         'Maximum Number of Consecutive Dry Days': number_max,
#         'Year': number_max_date,

#         'Mean Number of Consecutive Dry Days': number_mean,
        
#     }, index=['Stats'])
#     df.index.name = ' '

#     return np.round(df, 2)





def style_matrix(df_metrics, title = "Key Metrics Summary"):
    """
    Format and validate the metrics DataFrame for display.
    """

    if not isinstance(df_metrics, pd.DataFrame):
        raise TypeError("df_metrics must be a pandas DataFrame")

    required_cols = {"Metric", "Value"}
    if not required_cols.issubset(set(df_metrics.columns)):
        raise ValueError('df_metrics must contain "Metric" and "Value" columns')

    styled = (
        df_metrics.style
        .hide(axis="index")
        .set_caption(title)
        .set_table_styles([
            {
                'selector': 'caption',
                'props': [
                    ('font-size', '18px'),
                    ('font-weight', 'bold'),
                    ('text-align', 'center'),
                    ('color', '#2C013B'),
                    ('margin-bottom', '12px')   # más espacio
                ]
            },
            {
                'selector': 'thead th',
                'props': [
                    ('background-color', "#eed3f4"),
                    ('color',  "#2C013B"),
                    ('font-weight', 'bold'),
                    ('text-align', 'center')
                ]
            },
            {
                'selector': 'td',
                'props': [
                    ('padding', '6px'),
                    ('border-bottom', '1px solid #ccc'),
                    ('text-align', 'center')
                ]
            },
        ])
        .set_properties(subset=["Metric"], **{"text-align": "center"})
        .applymap(lambda v: "font-weight: bold", subset=["Metric"])
    )


    # 🔥 Final formatting applied here
    return styled.format({
        "Value": "{:.3f}",
        "Year": "{:.0f}"     # <- Sin decimales
    }, na_rep="")



## CO2 TABLES ##

def table_co2(MLO_data, trend):

    data = MLO_data.dropna()

    metrics = {
        "Metric": [
            "Mean CO2 Concentration (ppm)",
            "Change in CO2 Concentration since 1951 (ppm)",
            "Rate of Change in CO2 Concentration (ppm/year)",   
        ],
        "Value": [
            data.CO2.mean(),
            trend[0] * (data.index.year[-1] - 1951),
            trend[0],

        ]
    }

    df_metrics = pd.DataFrame(metrics)
    return df_metrics

## TEMPERATURE TABLES ##

def table_temp_11(df1, trend):
    mean_temp = df1.tmean.mean()
    change_annual_mean_temp = trend * (df1.index.year[-1] - 1951)
    rate_change = trend

    metrics = {
        "Metric": [
            "Mean Temperature",
            "Change in Annual Mean Temperature since 1951",
            "Rate of Change (°C/year)"
        ],
        "Value": [
            mean_temp,
            change_annual_mean_temp,
            rate_change
        ]
    }

    df_metrics = pd.DataFrame(metrics)
    return df_metrics



def table_temp_12(st_data, trend_maximum, trend_minimum):

    metrics = {
        "Metric": [
            "Annual Maximum Temperature (°C)",
            "Change in Annual Maximum Temperature since 1951",
            "Rate of Change in Annual Maximum Temperature (°C/year)",
            "Annual Minimum Temperature (°C)",
            "Change in Annual Minimum Temperature since 1951",
            "Rate of Change in Annual Minimum Temperature (°C/year)"
        ],
        "Value": [
            st_data.TMAX.max(),
            trend_maximum * (st_data.index.year[-1] - 1951),
            trend_maximum,

            st_data.TMIN.min(),
            trend_minimum * (st_data.index.year[-1] - 1951),
            trend_minimum,
        ]
    }

    df_metrics = pd.DataFrame(metrics)
    return df_metrics



def table_temp_13(st_data, annual_hot, annual_cold, df_hot_anom, df_cold_anom, TRENDS):

    # hot_cold_days

    metrics = {
        "Metric": [
            "Daily Maximum Temperature (°C)",
            "Daily Minimum Temperature (°C)",
            " ",
            'Average number of hot days',
            'Change in Average Annual Number of Hot Days',
            "Average Annual Number of Hot days: 1990-2000",
            "Average Annual Number of Hot days: 2000-2010",
            "Average Annual Number of Hot days: 2010-2020",
            "Maximum number of hot days",
            " ",

            'Average number of cold nights',
            'Change in Average Annual Number of Cold Nights',
            "Average Annual Number of Cold Nights: 1990-2000",
            "Average Annual Number of Cold Nights: 2000-2010",
            "Average Annual Number of Cold Nights: 2010-2020",
            "Maximum number of cold nights",
            " ",

        ],
        "Value": [
            st_data.TMAX.max(),
            st_data.TMIN.min(),
            
            np.nan,

            np.nanmean(df_hot_anom*3.6525),
            TRENDS[1],
            np.nanmean(df_hot_anom.loc["1990":"2000"]*3.6525),
            np.nanmean(df_hot_anom.loc["2000":"2010"]*3.6525),
            np.nanmean(df_hot_anom.loc["2010":"2020"]*3.6525),
            int(np.nanmax(annual_hot)),
            np.nan,


            np.nanmean(df_cold_anom*3.6525),
            TRENDS[0],
            np.nanmean(df_cold_anom.loc["1990":"2000"]*3.6525),
            np.nanmean(df_cold_anom.loc["2000":"2010"]*3.6525),
            np.nanmean(df_cold_anom.loc["2010":"2020"]*3.6525),
            int(np.nanmax(annual_cold)),
            np.nan,
        ],
        "Year": [
            np.nan,
            np.nan,
            
            np.nan,

            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            int(annual_hot['Perc_Anom'].idxmax().year),
            np.nan,


            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            int(annual_cold['Perc_Anom'].idxmax().year),
            np.nan,
        ]
    }

    df_metrics = pd.DataFrame(metrics)
    df_metrics

    return df_metrics


## RAINFALL TABLES ##

def table_rain_21(data, trend_da_mean, trend_ac_an):

    data = data.drop('season', axis = 1)
    datag = (data.groupby(data.index.year).sum()/ data.groupby(data.index.year).count()) * 365
    datag.index = pd.to_datetime(datag.index, format = '%Y')

    metrics = {
        "Metric": [
            "Daily Precipitation Mean (mm)",
            "Change in Daily Precipitation since 1951 (mm)",
            "Rate of Change in Daily Precipitation (mm/year)",
            " ",
            "Mean Accumulated Annual Precipitation (mm)",
            "Change in Accumulated Annual Precipitation since 1951 (mm)",
            "Rate of Change in Accumulated Annual Precipitation (mm/year)",

        ],
        "Value": [
            data.PRCP.mean(),
            trend_da_mean[0] * (data.index.year[-1] - 1951) * 365,
            trend_da_mean[0],

            np.nan,
            datag.PRCP.mean(),
            
            trend_ac_an * (datag.index.year[-1] - 1951),
            trend_ac_an,
        ]
    }

    df_metrics = pd.DataFrame(metrics)
    return df_metrics

def table_rain_22(data, trend_dry_days, trend_max_ndays):

    data_dry = data.loc[data.below_threshold == True]


    metrics = {
        "Metric": [
            "Annual Average Number of Dry Days [<1mm]",
            "Change in Number of Dry Days since 1951",
            "Rate of Change in Number of Dry Days (days/year)",
            "Average Number of Dry Days: 1951 - 1962",
            "Average Number of Dry Days: 2012 - 2021",
            f"Number of dry days in the driest year on record ({data_dry.groupby(data_dry.index.year).count().iloc[data_dry.PRCP.groupby(data_dry.index.year).count().argmax()].name})",
            " ",

            "Maximum number of consecutive dry days on record",
            "Change in Maximum Consecutive Dry Days since 1951",
            "Rate of Change in Maximum Consecutive Dry Days (days/year)",
            "Average Annual Maximum Consecutive Dry Days: 1951 - 1962",
            "Average Annual Maximum Consecutive Dry Days: 2012 - 2021",


        ],
        "Value": [
            data.loc[data['wet_day_t'] == 0].groupby(data.loc[data['wet_day_t'] == 0].index.year).PRCP.count().mean(),
            trend_dry_days * (data_dry.index.year.max() - 1951),
            trend_dry_days,
            data_dry.PRCP.loc['1951':'1962'].groupby(data_dry.loc['1951':'1962'].index.year).count().mean(),
            data_dry.PRCP.loc['2012':'2021'].groupby(data_dry.loc['2012':'2021'].index.year).count().mean(),
            data_dry.groupby(data_dry.index.year).count().iloc[data_dry.PRCP.groupby(data_dry.index.year).count().argmax()].PRCP,
            np.nan,

            data.groupby(data.index.year)['consecutive_days'].max().max(),
            trend_max_ndays * (data.index.year.max() - 1951),
            trend_max_ndays,
            data.groupby(data.index.year)['consecutive_days'].max().loc['1951':'1962'].mean(),
            data.groupby(data.index.year)['consecutive_days'].max().loc['2012':'2021'].mean()
        ]
    }

    df_metrics = pd.DataFrame(metrics)
    return df_metrics

def table_rain_23(data_th_1mm, data_th_95, trend_wet, trend_95):

    data_wet = data_th_1mm.loc[data_th_1mm.wet_day == 1]

    metrics = {
        "Metric": [
            "Annual average of wet days",
            "Change in number of wet days from 1951",
            "Rate of change in number of wet days",
            "Average Number of Wet Days: 1951 - 1962",
            "Average Number of Wet Days: 2012 - 2021",
            f"Wet days in the wettest year: {data_wet.groupby(data_wet.index.year).count().iloc[data_wet.groupby(data_wet.index.year).count().PRCP.argmax()].name}",
            " ",
            "Annual average number of days with heavy rainfall (>95th percentile)",
            "Change in number of heavy rainfall days from 1951",
            "Rate of change in number of heavy rainfall days",
            "Average Number of Heavy Rainfall Days: 1951 - 1962",
            "Average Number of Heavy Rainfall Days: 2012 - 2021",


        ],
        "Value": [
            data_th_1mm.loc[data_th_1mm['wet_day_t'] == 1].groupby(data_th_1mm.loc[data_th_1mm['wet_day_t'] == 1].index.year).PRCP.count().mean(),
            trend_wet * (data_wet.index.year.max() - 1951),
            trend_wet,
            data_wet.PRCP.loc['1951':'1962'].groupby(data_wet.loc['1951':'1962'].index.year).count().mean(),
            data_wet.PRCP.loc['2012':'2021'].groupby(data_wet.loc['2012':'2021'].index.year).count().mean(),
            data_wet.groupby(data_wet.index.year).count().iloc[data_wet.groupby(data_wet.index.year).count().PRCP.argmax()].PRCP,
            np.nan,

            data_th_95.loc[data_th_95['wet_day_t'] == 1].groupby(data_th_95.loc[data_th_95['wet_day_t'] == 1].index.year).PRCP.count().mean(),
            trend_95 * (data_th_95.index.year.max() - 1951),
            trend_95,
            data_th_95.loc[data_th_95['wet_day_t'] == 1].loc['1951':'1962'].groupby(data_th_95.loc[data_th_95['wet_day_t'] == 1].loc['1951':'1962'].index.year).PRCP.count().mean(),
            data_th_95.loc[data_th_95['wet_day_t'] == 1].loc['2012':'2021'].groupby(data_th_95.loc[data_th_95['wet_day_t'] == 1].loc['2012':'2021'].index.year).PRCP.count().mean(),

        ]
    }

    df_metrics = pd.DataFrame(metrics)
    return df_metrics

## TROPICAL CYCLONE TABLES ##

def table_tcs_32a(tcs_sel_params, oni):

    #Palau
    
    tcs_ratio_nino = len(tcs_sel_params.where(tcs_sel_params.oni_cat == 1, drop = True).storm)/len(oni.loc[oni.oni_cat == 1].index.year.unique())
    tcs_ratio_nino_severe = len(tcs_sel_params.where((tcs_sel_params.oni_cat == 1) & (tcs_sel_params.category >= 3), drop = True).storm)/\
                            len(oni.loc[oni.oni_cat == 1].index.year.unique())
    
    tcs_ratio_nina = len(tcs_sel_params.where(tcs_sel_params.oni_cat == -1, drop = True).storm)/len(oni.loc[oni.oni_cat == -1].index.year.unique())
    tcs_ratio_nina_severe = len(tcs_sel_params.where((tcs_sel_params.oni_cat == -1) & (tcs_sel_params.category >= 3), drop = True).storm)/\
                            len(oni.loc[oni.oni_cat == -1].index.year.unique())
    
    tcs_ratio_neutral = len(tcs_sel_params.where(tcs_sel_params.oni_cat == 0, drop = True).storm)/len(oni.loc[oni.oni_cat == 0].index.year.unique())
    tcs_ratio_neutral_severe = len(tcs_sel_params.where((tcs_sel_params.oni_cat == 0) & (tcs_sel_params.category >= 3), drop = True).storm)/\
                                len(oni.loc[oni.oni_cat == 0].index.year.unique())
    
    
    metrics = {
        "Metric": [
            " ",
            "VICINITY OF PALAU",
            " ",
            "Total number of tracks",
            "Tropical Storms per year",
            "Major Hurricanes (Category 3+) per year",
            " ",
            "EL NIÑO",
            "Total number of storm per year",
            "Major Hurricanes (Category 3+) per year",
            " ",
            "LA NIÑA",
            "Total number of storm per year",
            "Major Hurricanes (Category 3+) per year",
            " ",    
            "NEUTRAL",
            "Total number of storm per year",
            "Major Hurricanes (Category 3+) per year",  
        ],
        "Value": [
            np.nan,
            np.nan,
            np.nan,
            len(tcs_sel_params.storm),
            len(tcs_sel_params.where(tcs_sel_params.category == 0, drop = True).storm) / len(np.unique(tcs_sel_params.dmin_date.dt.year)),
            len(tcs_sel_params.where(tcs_sel_params.category >= 3, drop = True).storm) / len(np.unique(tcs_sel_params.dmin_date.dt.year)),
            np.nan,
            np.nan,
            tcs_ratio_nino,
            tcs_ratio_nino_severe,
            np.nan,
            np.nan,
            tcs_ratio_nina,
            tcs_ratio_nina_severe,
            np.nan,
            np.nan,
            tcs_ratio_neutral,
            tcs_ratio_neutral_severe
        ]
    }

    df_metrics = pd.DataFrame(metrics)
    return df_metrics


def table_tcs_32b(tcs_WP, oni):
   
    # WP

    tcs_g_WP = pd.DataFrame(tcs_WP.isel(date_time = 0).time.values)
    tcs_g_WP.index = tcs_g_WP[0]
    tcs_g_WP.index = pd.DatetimeIndex(tcs_g_WP.index).to_period('M').to_timestamp() + pd.offsets.MonthBegin(0)
    tcs_g_WP['oni_cat'] = oni.oni_cat
    tcs_WP['oni_cat'] = (('storm'), tcs_g_WP['oni_cat'].values)

    tcs_ratio_nino_WP = len(tcs_WP.where(tcs_WP.oni_cat == 1, drop = True).storm)/len(oni.loc[oni.oni_cat == 1].index.year.unique())
    tcs_ratio_nino_severe_WP = len(tcs_WP.where((tcs_WP.oni_cat == 1) & (tcs_WP.category >= 3), drop = True).storm)/\
                            len(oni.loc[oni.oni_cat == 1].index.year.unique())

    tcs_ratio_nina_WP = len(tcs_WP.where(tcs_WP.oni_cat == -1, drop = True).storm)/len(oni.loc[oni.oni_cat == -1].index.year.unique())
    tcs_ratio_nina_severe_WP = len(tcs_WP.where((tcs_WP.oni_cat == -1) & (tcs_WP.category >= 3), drop = True).storm)/\
                            len(oni.loc[oni.oni_cat == -1].index.year.unique())

    tcs_ratio_neutral_WP = len(tcs_WP.where(tcs_WP.oni_cat == 0, drop = True).storm)/len(oni.loc[oni.oni_cat == 0].index.year.unique())
    tcs_ratio_neutral_severe_WP = len(tcs_WP.where((tcs_WP.oni_cat == 0) & (tcs_WP.category >= 3), drop = True).storm)/\
                                len(oni.loc[oni.oni_cat == 0].index.year.unique())

    metrics = {
        "Metric": [
            " ",
            "WEST PACIFIC BASIN",
            " ",
            "Total number of tracks",
            "Tropical Storms per year",
            "Major Hurricanes (Category 3+) per year",
            " ",
            "EL NIÑO",
            "Total number of storm per year",
            "Major Hurricanes (Category 3+) per year",
            " ",
            "LA NIÑA",
            "Total number of storm per year",
            "Major Hurricanes (Category 3+) per year",
            " ",    
            "NEUTRAL",
            "Total number of storm per year",
            "Major Hurricanes (Category 3+) per year",  
        ],
        "Value": [
            np.nan,
            np.nan,
            np.nan,
            len(tcs_WP.storm),
            len(tcs_WP.where(tcs_WP.category == 0, drop = True).storm)/len(np.unique(tcs_WP.time.dt.year)),
            len(tcs_WP.where(tcs_WP.category >= 3, drop = True).storm)/len(np.unique(tcs_WP.time.dt.year)),
            np.nan,
            np.nan,
            tcs_ratio_nino_WP,
            tcs_ratio_nino_severe_WP,
            np.nan,
            np.nan,
            tcs_ratio_nina_WP,
            tcs_ratio_nina_severe_WP,
            np.nan,
            np.nan,
            tcs_ratio_neutral_WP,
            tcs_ratio_neutral_severe_WP
        ]
    }

    df_metrics = pd.DataFrame(metrics)
    return df_metrics