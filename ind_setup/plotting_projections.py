import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


loc = [7.28, 134.47]

scenario_color = {
        "ssp126": "green",
        "ssp245": "blue",
        "ssp370": "orange",
        "ssp585": "red"
    }

color_incr = ["gold", "orange", "crimson"]

def crossing_year(years, series, threshold):
    """
    First year where series >= threshold.
    Returns None if never crossed.
    """
    mask = series >= threshold
    if mask.any():
        return years[np.argmax(mask)]
    return None

def plot_projection_with_thresholds(st_data, ds_global, ds_palau, var_analysis_obs, var_analysis_proj, reference_period, ref_temp, incr_temps, ssp_scenarios):

    # Offsets to avoid overlaps
    dx_mean,  dy_mean  = 0.0,  0.06
    dx_upper, dy_upper = 0.8,  0.05
    dx_lower, dy_lower = -0.8, -0.05

    # --------------------------------------------------
    # Figure
    # --------------------------------------------------
    fig, ax = plt.subplots(figsize=(15, 6))

    # --------------------------------------------------
    # Observations
    # --------------------------------------------------
    st_data[var_analysis_obs].plot(
        ax=ax,
        color="darkslateblue",
        linestyle="-",
        alpha = .9,
        # marker="o",
        label="Observed Koror"
    )

    st_data.sel(time=slice(reference_period[0], reference_period[1]))[var_analysis_obs].plot(
        ax=ax,
        color="darkslateblue",
        linestyle="-",
        lw = 8,
        alpha = .4,
        # marker="o",
        label="Observed Koror - reference period"
    )

    coef = np.polyfit(st_data.time.dt.year, st_data[var_analysis_obs], 1)
    trend_vals = np.polyval(coef, st_data.time.dt.year)
    ax.plot(
        st_data.time,
        trend_vals,
        color="mediumslateblue",
        linestyle="-.",
        linewidth=1,
        label="Observed trend"
    )

    # --------------------------------------------------
    # Scenarios
    # --------------------------------------------------
    for scenario in ssp_scenarios:

        ds_global_scenario = ds_global.sel(scenario=scenario)[[var_analysis_proj]]
        mean_glob = ds_global_scenario[var_analysis_proj].mean(dim="model")
        mean_glob.plot(
            ax=ax,
            color=scenario_color[scenario],
            linestyle=":",
            label=f"Global mean {scenario}"
        )

        ds = ds_palau.sel(scenario=scenario)[[var_analysis_proj]]

        mean = ds[var_analysis_proj].mean(dim="model")
        std  = ds[var_analysis_proj].std(dim="model")

        years = mean.time
        mean_vals  = mean.values
        upper_vals = (mean + std).values
        lower_vals = (mean - std).values

        # ---- Mean and spread ----
        mean.plot(
            ax=ax,
            color=scenario_color[scenario],
            label=f"Scenario: {scenario}"
        )

        ax.fill_between(
            years,
            upper_vals,
            lower_vals,
            color=scenario_color[scenario],
            alpha=0.12,
            edgecolor=None
        )

        # ---- Threshold crossings ----
        for incr in incr_temps:

            threshold = ref_temp + incr

            y_mean  = crossing_year(years, mean_vals,  threshold)
            y_upper = crossing_year(years, upper_vals, threshold)
            y_lower = crossing_year(years, lower_vals, threshold)

            # Mean crossing (●) — big, scenario color
            if y_mean is not None:
                ax.scatter(
                    y_mean, threshold,
                    color=scenario_color[scenario],
                    s=70,
                    marker="o",
                    zorder=7
                )
                ax.text(
                    y_mean,# + dx_mean,
                    threshold + dy_mean,
                    f"{y_mean.time.dt.year.values}",
                    fontsize=9,
                    fontweight="bold",
                    color=scenario_color[scenario],
                    ha="center",
                    va="bottom"
                )

            # Upper band crossing (▲) — grey
            if y_upper is not None:
                ax.scatter(
                    y_upper, threshold,
                    color=scenario_color[scenario],
                    s=35,
                    marker="^",
                    zorder=6
                )
                ax.text(
                    y_upper,# + dx_upper,
                    threshold + dy_upper,
                    f"{y_upper.time.dt.year.values}",
                    fontsize=7,
                    color="grey",
                    ha="left",
                    va="bottom"
                )

            # Lower band crossing (▼) — grey
            if y_lower is not None:
                ax.scatter(
                    y_lower, threshold,
                    color=scenario_color[scenario],
                    s=35,
                    marker="v",
                    zorder=6
                )
                ax.text(
                    y_lower,# + dx_lower,
                    threshold,
                    f"{y_lower.time.dt.year.values}",
                    fontsize=7,
                    color="grey",
                    ha="right",
                    va="top"
                )

    # --------------------------------------------------
    # Reference and thresholds
    # --------------------------------------------------
    ax.axhline(
        ref_temp,
        linestyle=":",
        color="grey",
        label = f"Reference ({reference_period[0]}-{reference_period[1]}): {ref_temp:.2f}",
    )


    for incr in incr_temps:
        ax.axhline(
            ref_temp + incr,
            linestyle="-.",
            color=color_incr[incr_temps.index(incr)],
            linewidth=1,
        )
        ax.text(
            st_data.time.min() + 1,
            ref_temp + incr + 0.03,
            f"+{incr}°C",
            fontsize=12,
            color=color_incr[incr_temps.index(incr)],
            ha="left",
            va="bottom"
            )

    # --------------------------------------------------
    # Final formatting
    # --------------------------------------------------
    ax.set_ylabel("Temperature (°C)")
    ax.set_xlabel("Year")
    ax.set_xlim(st_data.time.min(), ds_global.time.max())
    ax.set_title(f"lat = {loc[0]:.2f}°, lon = {loc[1]:.2f}°")
    ax.legend(ncol= 2,  loc='upper left')

    ax.axvline(st_data.sel(time='2014').time.values, linestyle='--', color='black', label='2015')

    plt.tight_layout()
    plt.show()



def plot_projection_global_palau(ds_global, ds_palau, var_analysis_proj, ssp_scenarios):

    for scenario in ssp_scenarios:
        # ==================================================
        # PREPARE ENSEMBLE TIME SERIES
        # ==================================================
        # Global mean per model
        tas_global_models = ds_global.sel(scenario=scenario)[var_analysis_proj]

        # Ensemble statistics
        tas_global_mean = tas_global_models.mean(dim="model")
        tas_global_std  = tas_global_models.std(dim="model")

        # Palau mean per model
        tas_palau_models = ds_palau.sel(scenario=scenario)[var_analysis_proj]
        tas_palau_mean   = tas_palau_models.mean(dim="model")
        tas_palau_std    = tas_palau_models.std(dim="model")

        years = tas_global_mean["time"].dt.year.values

        # ==================================================
        # DEFINE GLOBAL WARMING SCENARIOS (ENSEMBLE MEAN)
        # ==================================================
        scenarios = [2.0, 2.5, 3.0]
        scenario_info = {}

        for sc in scenarios:
            idx = np.where(tas_global_mean.values >= sc)[0][0]
            scenario_info[sc] = {
                "year": years[idx],
                "palau": tas_palau_mean.values[idx],
                "time": tas_global_mean.time[idx].values
            }

        # ==================================================
        # PLOT
        # ==================================================
        fig, ax = plt.subplots(figsize=(15, 5))

        # --------------------------------------------------
        # ENSEMBLE MEAN + STD SHADE
        # --------------------------------------------------
        ax.fill_between(
            tas_global_mean.time.values,
            tas_global_mean.values - tas_global_std.values,
            tas_global_mean.values + tas_global_std.values,
            color=scenario_color[scenario],
            alpha=0.1,
            label="Ensemble ±1σ",
            zorder=1
        )

        tas_global_mean.plot(
            ax=ax,
            label="Global ensemble mean",
            color=scenario_color[scenario],
            linewidth=3.0,
            linestyle=":",
            zorder=3
        )

        # Palau ensemble mean
        tas_palau_mean.plot(
            ax=ax,
            label="Palau ensemble mean",
            color=scenario_color[scenario],
            linewidth=2.5,
            alpha=0.95,
            zorder=3
        )

        ax.fill_between(
            tas_palau_mean.time.values,
            tas_palau_mean.values - tas_palau_std.values,
            tas_palau_mean.values + tas_palau_std.values,
            color=scenario_color[scenario],
            alpha=0.25,
            label="Ensemble ±1σ",
            zorder=1
        )

        # --------------------------------------------------
        # SCENARIO LINES AND ANNOTATIONS
        # --------------------------------------------------
        label_offsets = [0.45, -0.45, 0.45]

        for i, (sc, info) in enumerate(scenario_info.items()):
            year_sc  = info["year"]
            palau_sc = info["palau"]
            time_sc  = info["time"]
            offset   = label_offsets[i]

            # Horizontal line (global level)
            ax.axhline(
                sc,
                color="gray",
                linestyle="--",
                linewidth=1.2,
                alpha=0.6,
                zorder=0
            )

            # Vertical line (year reached)
            ax.axvline(
                time_sc,
                color="black",
                linestyle="-",
                linewidth=2.2,
                alpha=0.85,
                zorder=2
            )

            # Marker on Palau
            ax.plot(
                time_sc,
                palau_sc,
                marker="o",
                markersize=7,
                color="black",
                zorder=4
            )

            # Annotation
            ax.text(
                time_sc,
                palau_sc + offset,
                f"+{sc}°C Global → {year_sc}\nPalau: {palau_sc:.2f}°C",
                fontsize=10,
                ha="left",
                va="center",
                bbox=dict(
                    boxstyle="round,pad=0.25",
                    fc="white",
                    ec="0.7",
                    alpha=0.9
                )
            )

        # --------------------------------------------------
        # FORMATTING
        # --------------------------------------------------
        ax.set_title(
            f"Global Warming Levels and Corresponding Local Warming in Palau\n"
            f"Ensemble mean ±1σ ({scenario})",
            fontsize=14,
            fontweight="bold",
            pad=10
        )

        ax.set_ylabel("Temperature anomaly (°C)", fontsize=12)
        ax.set_xlabel("")
        ax.set_xlim(tas_global_mean.time.min(), tas_global_mean.time.max())

        ax.grid(which="major", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.grid(which="minor", linestyle=":", linewidth=0.5, alpha=0.3)
        ax.minorticks_on()

        ax.legend(
            loc="upper left",
            frameon=True,
            framealpha=0.95,
            edgecolor="0.8"
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()



def get_box_data(ds_palau, scenario_info, incr_temps, ssp_scenarios, var_analysis):

    # ==================================================
    # PREPARE DATA FOR BOXPLOT
    # ==================================================
    box_data = []
    box_scenarios = []

    for wl in incr_temps:
        for scenario_analysis in ssp_scenarios:
            year = scenario_info[scenario_analysis][wl]["year"]

            values = (
                ds_palau
                .sel(scenario=scenario_analysis)
                .sel(time=year, method="nearest")[var_analysis]
                .values  # (model,)
            )

            box_data.append(values)
            box_scenarios.append(scenario_analysis)

    # ==================================================
    # DEFINE BOX POSITIONS (GROUPED BY WARMING LEVEL)
    # ==================================================
    positions = []
    pos = 1.0
    gap = 1.0
    offset = 0.35

    for _ in incr_temps:
        positions.extend([pos, pos + offset])
        pos += offset + gap

    return box_data, box_scenarios, positions



def plot_global_warming_boxplot(box_data, ds_palau,positions, box_scenarios, scenario_info, incr_temps, var_analysis, plot_ref = True):

    # ==================================================
    # PLOT — OPTION A (colored scenario years)
    # ==================================================
    fig, ax = plt.subplots(figsize=(10, 5))

    bp = ax.boxplot(
        box_data,
        positions=positions,
        widths=0.3,
        patch_artist=True,
        medianprops=dict(linewidth=2)
    )
    if plot_ref:
        ax.hlines(ds_palau.reference.values, 0.5, 4.5, lw = 1, colors=['orange', 'crimson'], linestyles='dashed', label='Reference (2015-2035)')

    # ---- Color boxes
    for i, (patch, scenario) in enumerate(zip(bp['boxes'], box_scenarios)):
        color = scenario_color[scenario]
        patch.set_facecolor(color)
        patch.set_alpha(0.3)
        bp['medians'][i].set_color(color)
        bp['medians'][i].set_linewidth(2)

    # ---- X-axis ticks (ONLY warming level)
    group_centers = [
        np.mean(positions[i*2:(i+1)*2])
        for i in range(len(incr_temps))
    ]

    ax.set_xticks(group_centers)
    ax.set_xticklabels([f"+{wl}°C" for wl in incr_temps], fontsize=12)

    # ---- Add colored scenario-year labels under ticks
    ymin, ymax = ax.get_ylim()
    y_text_1 = ymin - 0.10 * (ymax - ymin)
    y_text_2 = ymin - 0.16 * (ymax - ymin)

    for i, wl in enumerate(incr_temps):
        x = group_centers[i]

        y370 = np.datetime_as_string(
            scenario_info['ssp370'][wl]['year'], unit='Y'
        )
        y585 = np.datetime_as_string(
            scenario_info['ssp585'][wl]['year'], unit='Y'
        )

        ax.text(
            x, y_text_1,
            f"ssp370: {y370}",
            ha="center",
            va="top",
            fontsize=12,
            color=scenario_color['ssp370']
        )

        ax.text(
            x, y_text_2,
            f"ssp585: {y585}",
            ha="center",
            va="top",
            fontsize=12,
            color=scenario_color['ssp585']
        )

    # ---- Formatting
    ax.set_ylabel(var_analysis, color="0.35")
    ax.set_title("Palau response at global warming levels", color="0.25")

    legend_elements = [
        Patch(facecolor='orange', edgecolor='k', alpha=0.5, label='ssp370'),
        Patch(facecolor='crimson', edgecolor='k', alpha=0.5, label='ssp585')
    ]
    ax.legend(handles=legend_elements, ncol = 2, title="Scenario")

    # ---- Gray spines
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("0.6")
        spine.set_linewidth(1.0)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # ---- Gray ticks and tick labels
    ax.tick_params(
        axis="both",
        which="both",
        colors="0.5",
        labelsize=12
    )

    ax.grid(axis='y', alpha=0.3, color="0.7")

    plt.tight_layout()

    return fig, ax



import matplotlib.pyplot as plt


def plot_palau_temperature_anomalies_by_scenario(
    ds_palau,
    scenario,
    baseline_start="2015",
    baseline_end="2035",
    figsize=(15, 6) 
    ):
    """
    Plot Palau temperature anomalies (tas, tasmin, tasmax) for a single scenario.
    Styling (spines, labels, titles) is colored according to the scenario.
    """

    # ---------------------------------
    # Scenario-specific styling
    # ---------------------------------
    scenario_style = {
        "ssp370": {
            "color": "orange",
            "linestyle": ":"
        },
        "ssp585": {
            "color": "crimson",
            "linestyle": "-"
        }
    }

    var_color = {
        "tas": "grey",
        "tasmin": "royalblue",
        "tasmax": "crimson"
    }

    scen_color = scenario_style[scenario]["color"]
    lt = scenario_style[scenario]["linestyle"]

    # ---------------------------------
    # Figure
    # ---------------------------------
    fig, ax = plt.subplots(figsize=figsize)

    # ---------------------------------
    # Plot variables
    # ---------------------------------
    for var in ["tas", "tasmin", "tasmax"]:

        da = ds_palau[var].sel(scenario=scenario)

        # Baseline per model
        baseline = da.sel(
            time=slice(baseline_start, baseline_end)
        ).mean(dim="time")

        # Anomalies
        da_anom = da - baseline

        # Percentiles across models
        p10 = da_anom.quantile(0.10, dim="model")
        p25 = da_anom.quantile(0.25, dim="model")
        p50 = da_anom.quantile(0.50, dim="model")
        p75 = da_anom.quantile(0.75, dim="model")
        p90 = da_anom.quantile(0.90, dim="model")

        color = var_color[var]

        # Uncertainty bands
        ax.fill_between(
            da_anom["time"],
            p10,
            p90,
            color=color,
            alpha=0.08,
            linewidth=0
        )

        ax.fill_between(
            da_anom["time"],
            p25,
            p75,
            color=color,
            alpha=0.14,
            linewidth=0
        )

        # Median line
        ax.plot(
            da_anom["time"],
            p50,
            color=color,
            linewidth=1.6 if var == "tas" else 1.3,
            linestyle=lt,
            label=var
        )

    # Zero anomaly reference
    ax.axhline(
        0,
        color=scen_color,
        linestyle="--",
        linewidth=1.2,
        alpha=0.7
    )

    # ---------------------------------
    # Styling (scenario-colored chrome)
    # ---------------------------------
    ax.set_title(
        f"Palau – {scenario}\nTemperature anomalies vs {baseline_start}–{baseline_end}",
        fontsize=15,
        color=scen_color,
        pad=12
    )

    ax.set_xlim(ds_palau.time.min(), ds_palau.time.max())
    ax.set_xlabel("Year", fontsize=12, color=scen_color)
    ax.set_ylabel("Temperature anomaly (°C)", fontsize=12, color=scen_color)

    # Spines
    for spine in ax.spines.values():
        spine.set_color(scen_color)
        spine.set_linewidth(1.2)

    # Ticks
    ax.tick_params(
        axis="both",
        colors=scen_color,
        labelsize=11
    )

    # Grid
    ax.grid(True, alpha=0.3, color=scen_color)

    # Legend (neutral)
    ax.legend(
        title="Variable",
        fontsize=11,
        title_fontsize=12,
        frameon=True,
        edgecolor="0.7"
    )

    plt.tight_layout()
    plt.show()


def plot_violin_by_warming_level(
    ds_palau,
    scenario_info,
    var,
    scenarios=("ssp370", "ssp585"),
    warming_levels=(2.0, 2.5, 3.0),
    scenario_color=None,
    alpha_val=0.35,
    group_gap=1.5,
    violin_width=0.8,
    figsize=(15, 4),
    plot_ref=True,
):
    """
    Violin plot of inter-model spread grouped by global warming levels.
    Adds scenario-specific years below each warming level.
    """

    if scenario_color is None:
        scenario_color = {
            "ssp370": "orange",
            "ssp585": "crimson",
        }

    # ---------------------------------
    # PREPARE DATA
    # ---------------------------------
    data = []
    positions = []
    colors = []

    pos = 1

    for wl in warming_levels:
        for sc in scenarios:

            year = scenario_info[sc][wl]["year"]

            da = (
                ds_palau[var]
                .sel(scenario=sc)
                .sel(time=year, method="nearest")
            )

            vals = da.dropna(dim="model").values
            if len(vals) == 0:
                vals = np.array([np.nan])

            data.append(vals)
            positions.append(pos)
            colors.append(scenario_color[sc])

            pos += 1

        pos += group_gap

    # ---------------------------------
    # PLOT
    # ---------------------------------
    fig, ax = plt.subplots(figsize=figsize)

    vp = ax.violinplot(
        data,
        positions=positions,
        widths=violin_width,
        showmeans=False,
        showmedians=True,
        showextrema=True,
    )

    # ---- Color violins
    for body, color in zip(vp["bodies"], colors):
        body.set_facecolor(color)
        body.set_alpha(alpha_val)
        body.set_edgecolor("black")
        body.set_linewidth(0.5)

    # ---- Style median and extrema
    vp["cmedians"].set_color("black")
    vp["cmedians"].set_linewidth(2)

    for part in ["cbars", "cmins", "cmaxes"]:
        vp[part].set_color("black")
        vp[part].set_linewidth(0.8)

    # ---------------------------------
    # REFERENCE LINES (per scenario)
    # ---------------------------------
    ref_handles = []

    if plot_ref:

        ref_period = slice("2015-01-01", "2035-12-31")
        xmin, xmax = ax.get_xlim()

        for sc in scenarios:

            reference = (
                ds_palau[var]
                .sel(scenario=sc)
                .sel(time=ref_period)
                .mean(["time", "model"])
                .values
            )

            ax.hlines(
                reference,
                xmin,
                xmax,
                lw=.8,
                colors=scenario_color[sc],
                linestyles="dashed",
            )

            ref_handles.append(
                Line2D(
                    [0], [0],
                    color=scenario_color[sc],
                    lw=1.8,
                    linestyle="dashed",
                    label=f"{sc} reference (2015–2035)",
                )
            )
        ax.set_xlim(xmin, xmax)

    # ---------------------------------
    # AXES & LABELS
    # ---------------------------------
    tick_positions = []
    start = 1
    for _ in warming_levels:
        tick_positions.append(start + (len(scenarios) - 1) / 2)
        start += len(scenarios) + group_gap

    ax.set_xticks(tick_positions)
    ax.set_xticklabels([f"+{wl}°C" for wl in warming_levels], fontsize=11)

    ax.set_xlabel("Global warming level", labelpad=40, fontsize=12)
    ax.set_ylabel(var.replace("_", " ").title(), fontsize=12)

    
    ax.set_title(
        f"{var.replace('_', ' ').title()} – Inter-model spread\n"
        "Grouped by scenario at global warming levels"
    )

    # ---- Scenario years under ticks
    ymin, ymax = ax.get_ylim()
    y_text_1 = ymin - 0.10 * (ymax - ymin)
    y_text_2 = ymin - 0.17 * (ymax - ymin)

    for i, wl in enumerate(warming_levels):
        x = tick_positions[i]

        for j, sc in enumerate(scenarios):
            year_str = np.datetime_as_string(
                scenario_info[sc][wl]["year"], unit="Y"
            )

            y_pos = y_text_1 if j == 0 else y_text_2

            ax.text(
                x,
                y_pos,
                f"{sc}: {year_str}",
                ha="center",
                va="top",
                fontsize=10,
                color=scenario_color[sc],
            )

    # ---------------------------------
    # LEGEND
    # ---------------------------------
    violin_handles = [
        Line2D(
            [0], [0],
            color=scenario_color[sc],
            lw=6,
            alpha=alpha_val,
            label=sc,
        )
        for sc in scenarios
    ]

    handles = violin_handles + ref_handles

    ax.legend(
        handles=handles,
        ncol=2,
        fontsize=11,
        frameon=False,
    )

    # ---------------------------------
    # STYLE
    # ---------------------------------
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_alpha(0.4)
    ax.spines["bottom"].set_alpha(0.4)

    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    return fig, ax