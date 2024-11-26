import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.patches import Circle
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

from .colors import plotting_style
plotting_style()

import contextily as ctx
from mpl_toolkits.axes_grid1 import make_axes_locatable
import plotly.graph_objects as go


def get_storm_linestyle(vel):
    if vel >= 60:
        return 'solid'
    elif vel >= 40:
        return 'dashed'
    elif vel >= 20:
        return 'dashdot'
    else:
        return 'dotted'

def get_storm_color(categ):

    dcs = {
        0 : 'green',
        1 : 'yellow',
        2 : 'orange',
        3 : 'red',
        4 : 'purple',
        5 : 'black',
    }

    return dcs[categ]


def Plot_TCs_HistoricalTracks_Category(xds_TCs_r1, cat,
                                       lon1, lon2, lat1, lat2,
                                       pnt_lon, pnt_lat, r1,
                                       nm_lon='lon', nm_lat='lat',
                                       show=True):
    """
    Plot Historical TCs category map using Cartopy.
    """

    # Define projection
    projection = ccrs.PlateCarree(central_longitude=180)

    # Create figure and axes with projection
    fig, ax = plt.subplots(1, figsize=(10, 8), subplot_kw={'projection': projection})

    # Add geographic features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=':')
    ax.add_feature(cfeature.LAND, color='silver')
    ax.add_feature(cfeature.OCEAN, color='lightcyan')

    # Set extent (bounding box)
    ax.set_extent([lon1, lon2, lat1, lat2], crs=ccrs.PlateCarree())

    # Add gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()

    # Plot storm tracks
    for s in range(len(xds_TCs_r1.storm)):
        lon = xds_TCs_r1.isel(storm=s)[nm_lon].values[:]
        lon[lon < 0] += 360  # Convert to 0-360 if needed

        # Plot storm track
        ax.plot(
            lon, xds_TCs_r1.isel(storm=s)[nm_lat].values[:],
            '-', color=get_storm_color(int(cat[s].values)),
            alpha=0.5, transform=ccrs.PlateCarree()
        )

        # Mark storm start points
        ax.plot(
            lon[0], xds_TCs_r1.isel(storm=s)[nm_lat].values[0],
            '.', color=get_storm_color(int(cat[s].values)),
            markersize=10, transform=ccrs.PlateCarree()
        )

    # Plot study site
    ax.plot(
        pnt_lon, pnt_lat, '.', color='brown',
        markersize=15, label='STUDY SITE', transform=ccrs.PlateCarree()
    )

    # Plot circle around the study site
    circle = Circle(
        (pnt_lon, pnt_lat), r1,
        facecolor='grey', edgecolor='grey',
        linewidth=3, alpha=0.5, transform=ccrs.PlateCarree()
    )
    ax.add_patch(circle)

    # Customize plot
    ax.set_title('Historical TCs', fontsize=15)
    ax.legend(loc='lower left', fontsize=12)
    ax.set_aspect('equal')  # Allow automatic scaling for map aspect ratio
    return ax


## Interactive Plotting

class Map(object):

	def __init__(self):

		self.bathy = None
		self.dem = None

	def attrs_axes(self, ax, extent):

		ax.set_xlabel('X UTM [m]', color='dimgray',  fontweight='bold')
		ax.set_ylabel('Y UTM [m]', color='dimgray',  fontweight='bold')
		ax.tick_params(axis='both', which='major', labelcolor='dimgray')
		#ax.set_title('')
		ax.set_aspect('equal')
		ax.set_xlim(extent[0], extent[1])
		ax.set_ylim(extent[2], extent[3])

	def attrs_axes_empty(self, ax, extent):

		ax.set_xlabel('')
		ax.set_ylabel('')
		ax.set_xticks([])
		ax.set_yticks([])
		ax.set_title('')
		ax.set_xlim(extent[0], extent[1])
		ax.set_ylim(extent[2], extent[3])
		
	def add_colorbar_composed(self, fig, ax, vmin, vmax, mymap, title):

		norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
		sm = plt.cm.ScalarMappable(cmap=mymap, norm=norm)
		sm.set_array([])

		ax_divider = make_axes_locatable(ax)
		cax = ax_divider.append_axes("right", size="3%", pad="2%")
		cbar = fig.colorbar(sm, cax=cax, orientation="vertical")
		cbar.set_label(title, fontweight='bold')
		cbar.outline.set_visible(False)

	def add_colorbar_loss(self, fig, ax, sm, mymap, title):

		ax_divider = make_axes_locatable(ax)
		cax = ax_divider.append_axes("right", size="3%", pad="2%")
		cbar = fig.colorbar(sm, cax=cax, orientation="vertical",  format='%d')
		cbar.set_label(title, fontweight='bold')
		#cbar.outline.set_visible(False)


	def ocean_land_cmap(self, cmap_ocean, cmap_land, vmin, vmax, min_p_color):

		colors1 = cmap_land(np.linspace(0, 0.9, np.abs(vmax)))
		colors2= cmap_ocean(np.linspace(min_p_color, 1, np.abs(vmin)))

		# combine them and build a new colormap
		c12 = np.vstack((colors2, colors1))
		mymap = mcolors.LinearSegmentedColormap.from_list('my_colormap', c12)
		mymap1 = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors1)
		mymap2 = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors2)        
       
		return(mymap, mymap1, mymap2)

	def Background_island_capital(self, island_extent, capital_extent, title=None):
		
		fig, ax = plt.subplots(1, 1, figsize=(20, 7), sharex=True, sharey=True)
		axz = ax.inset_axes([0.55, 0.65, 0.4, 0.3])
		_ = self.BackGround_2cmaps(fig, ax, island_extent, plt.cm.Blues_r, cmocean.cm.turbid,  [], [], vmin=-2000, vmax=1000, cbar=False, alpha=0.9)

		_ = self.BackGround_2cmaps(fig, axz, capital_extent, plt.cm.Blues_r, cmocean.cm.turbid,  [], [], vmin=-2000, vmax=1000, cbar=False, alpha=0.9, attrs=False)

		self.attrs_axes(ax, island_extent)
		self.attrs_axes_empty(axz, capital_extent)

		ax.indicate_inset_zoom(axz, edgecolor="black")
		ax.set_title(title)
		return(fig, ax, axz)
		
	def BackGround_2cmaps(self, fig, ax, extent, cmap_ocean, cmap_land,  ext1=None, ext2=None, vmin=None, vmax=None, cbar=True, alpha=0.5, min_p_color=0.5, attrs=True):
		'''
		args: 
         site: 
      	ext1, ext2: matplotlib.patches to indicate zoomed area

		return:
         ax object
		'''
	
		bathy = - self.bathy
		dem = self.dem
        
		mymap, mymap1, mymap2 = self.ocean_land_cmap(cmap_ocean, cmap_land, vmin, vmax, min_p_color)
      
		# hillshade = es.hillshade(bathy, azimuth=140, altitude=40)
		im = ax.pcolorfast(bathy.x, bathy.y, bathy, cmap=mymap2, norm=mcolors.SymLogNorm(linthresh=1, vmin=vmin, vmax=0), alpha=alpha, zorder=1)
		# ax.pcolorfast(bathy.x, bathy.y, hillshade, cmap="Greys", norm=mcolors.SymLogNorm(linthresh=1), alpha=0.05, zorder=2)

		# hillshade = es.hillshade(dem.z, azimuth=0, altitude=40)
		im = ax.pcolorfast(dem.x, dem.y, dem.z, cmap=mymap1, norm=mcolors.SymLogNorm(linthresh=1, vmin=0, vmax=vmax), alpha=0.8, zorder=3)
		# ax.pcolorfast(dem.x, dem.y, hillshade, cmap="Greys", norm=mcolors.SymLogNorm(linthresh=1), alpha=0.05, zorder=4)

		if attrs:
			self.attrs_axes(ax, extent)

		if ext1: ax.plot([ext1[0], ext1[1], ext1[1], ext1[0], ext1[0]], [ext1[2], ext1[2], ext1[3], ext1[3], ext1[2]], linewidth=2, c='k',  linestyle='--', zorder=6)
		if ext2: ax.plot([ext2[0], ext2[1], ext2[1], ext2[0], ext2[0]], [ext2[2], ext2[2], ext2[3], ext2[3], ext2[2]], linewidth=2, c='k',  linestyle='--', zorder=7)

		if cbar: self.add_colorbar_composed(fig, ax, vmin, vmax, mymap, title='Depth / Elevation (m)')

		return(ax)


	def BackGround_1cmaps(self, fig, ax, extent, cmap_land, code_utm, cbar=False, vmin=None, vmax=None, alpha=0.5, source=ctx.providers.OpenTopoMap):
		'''
      args: 
          ext1, ext2: matplotlib.patches to indicate zoomed area

      return:
          ax object
      '''
        
		dem = self.dem

		mymap, mymap1, mymap2 = self.ocean_land_cmap(cmap_land, cmap_land, vmin, vmax, 0)
        
		# hillshade = es.hillshade(dem.z, azimuth=0, altitude=40)
		im = ax.pcolormesh(dem.x, dem.y, dem.z, cmap=mymap1, norm=mcolors.SymLogNorm(linthresh=1, vmin=0, vmax=vmax), alpha=alpha, zorder=2)
		# ax.pcolorfast(dem.x, dem.y, hillshade, cmap="Greys", norm=mcolors.SymLogNorm(linthresh=1), alpha=0.1, zorder=3)

		ax.set_xlim(extent[0], extent[1])
		ax.set_ylim(extent[2], extent[3])
		self.attrs_axes(ax)

		ctx.add_basemap(ax, zoom = 10, crs='epsg:{0}'.format(code_utm), source=source, attribution=False, zorder=1)

		if cbar: self.add_colorbar_composed(fig, ax, vmin, vmax, mymap, title='Depth / Elevation (m)')
			
		return(ax)
	
	def tcs_plotly_syn(self, ds_data_tcs, lon_site, lat_site):
	
		fig = go.FigureWidget()

		for st in ds_data_tcs.storm:

			df = ds_data_tcs.sel(storm=st).to_dataframe().dropna(subset=['lon'])
			if len(df)> 0:
				print(df)

				fig.add_trace(go.Scattermapbox(
					mode = "markers+lines",
					lon = np.round(df.ylon_TC.values, 2),
					lat = np.round(df.ylat_TC.values, 2),
					name = 'ID {0}'.format(df.storm.values[0]),
			))

				fig.update_layout(legend=dict(orientation='v', yanchor='top', xanchor='right'))

				fig.update_layout(mapbox_style="open-street-map",
							width=1000,
							height=700,
							mapbox=dict(
								bearing=0,
								center=go.layout.mapbox.Center(
									lat=lat_site,
									lon=lon_site,
								),
								pitch=0,
								zoom=4
							),
					margin={"r":0, "t":0, "l":0, "b":0},
					showlegend=True,
					)
		return(fig)

	def tcs_plotly(self, ds_data_tcs, lon_site, lat_site, color=False):
	
		fig = go.FigureWidget()

		for st in ds_data_tcs.storm:

			df = ds_data_tcs.sel(storm=st).to_dataframe().dropna(subset=['lon'])
			if len(df) > 0:
				if color:
					fig.add_trace(go.Scattermapbox(
						mode = "markers+lines",
						lon = np.round(df.lon.values, 2),
						lat = np.round(df.lat.values, 2),
						name = '{0} {1}'.format(df.name.values[0], int(df.year.values[0])),
						text = ['Date: {0} \n Wind: {1} \n Pressure:  {2}'.format(df.loc[p].time, df.loc[p].wmo_wind, df.loc[p].wmo_pres) for p in df.index],
						marker = dict(size=15, color=df['wmo_pres'], showscale=True, colorbar=dict(title= 'Minimun Central Pressure (mb)')),
				))
				else:
					fig.add_trace(go.Scattermapbox(
						mode = "markers+lines",
						lon = np.round(df.lon.values, 2),
						lat = np.round(df.lat.values, 2),
						name = '{0} {1}'.format(df.name.values[0], int(df.year.values[0])),
				))

				fig.update_layout(legend=dict(orientation='v', yanchor='top', xanchor='right'))

				fig.update_layout(mapbox_style="open-street-map",
							width=1000,
							height=700,
							mapbox=dict(
								bearing=0,
								center=go.layout.mapbox.Center(
									lat=lat_site,
									lon=lon_site,
								),
								pitch=0,
								zoom=4
							),
					margin={"r":0, "t":0, "l":0, "b":0},
					showlegend=True,
					)
		return(fig)