---
layout: post
title:  "Analyzing Forest Fires in Northern Algeria during 2021 using NASA FIRMS data in Python"
date:   2021-09-15 11:31:00 +0100
categories: Julia Flux ML
---

In this post, I will be using [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov) data in Python. The goal is to get a reading on the surface on fire in each province as a function of time. I'll be introducing:

+ Loading SHP data.
+ Reprojecting the data.
+ Clipping map data.
+ Spatial Join.
+ Plotting spatial data in video format using matplotlib, GeoPandas and ffmpeg.
## Loading Data:
## Loading SHP file:

To get started, we get our data from the [NASA FIRMS Archive](https://firms.modaps.eosdis.nasa.gov/download) database, by creating a request for Algeria during the whole year of 2021 (I've used only VIIRS NOAA data here, you may use others but beware that the data may be different if you pick more than one). After a few minutes the request should be complete and you'll receive a zip file containing a Readme as well as 4 other files. Extract the 4 files into your working repository to load them.

Once the files are received and extracted, we'll need GeoPandas to use the data within Python:

{% highlight python %}
import geopandas as gpd
data_gdf = gpd.read_file("fire_nrt_J1V-C2_262461.shp")
{% endhighlight %}

the result, `data_gdf` is a GeoPandas _GeoDataFrame_. If you're familiar with Pandas, a GeoDataFrame is a normal Pandas DataFrame with a special column: `geometry`, this column stores the spatial geometric properties of each data point. In our case the geometry column stores the central point of the scanned surface. This `geometry` column is used by GeoPandas to perform spatial operations. _Note: I highly recommend skimming through the Helsinki university's [auto-gis](https://autogis-site.readthedocs.io/en/latest/) course for more information on geospatial tools in Python_

However, the resulting GeoDataFrame has many columns most of which are not needed. We will extract the columns that we need using:

{% highlight python %}
data_gdf.drop(data_gdf.columns.difference(['ACQ_DATE','ACQ_TIME', 'CONFIDENCE', 'SCAN', 'TRACK', 'geometry']), axis=1, inplace=True)
{% endhighlight %}

this only keeps  the columns we're interested in:

| Field          | Description                                             |
|----------------|---------------------------------------------------------|
| ACQ\_DATE      | Date of Acquisition of data                             |
| ACQ\_TIME      | Time of Acquisition of data                             |
| CONFIDENCE     | Confidence of fire in square n:w:Nominal, l:Low, h:High |
| SCAN and TRACK | Values Reflecting actual pixel size                     |
| geometry       | Geometry column                                         |

in addition, it's preferable to produce two extra columns `date` and `datetime` that will encode the date and timestamps respectively for the analysis,

{% highlight python %}
data_gdf['date'] = data_gdf.ACQ_DATE.apply(pd.Timestamp)
data_gdf['datetime'] = data_gdf.apply(lambda x: pd.Timestamp(f"{x['ACQ_DATE']}/{x['ACQ_TIME']}"), axis=1)
data_gdf.drop('ACQ_DATE', axis=1, inplace=True) #Not needed anymore
{% endhighlight %}

# Reprojecting the data:

The geometric data we have is in the WGS (degrees) coordinate system, while we can work with it, for our case it may be preferable to switch to Mercator projection to have better plots at the end,

{% highlight python %}
data_gdf = data_gdf.to_crs(3857)
{% endhighlight %}

the `3857` code is an [EPSG code](https://en.wikipedia.org/wiki/EPSG_Geodetic_Parameter_Dataset) for the Mercator projection.

_Note: In our case the CRS projection was optional, if our geometric data was given as a polygon directly the projection would be needed to calculate the surface_

# Getting pixel size:

Due to the nature of satellite imagery, pixels are not all the same size. Pixels that are taken vertically from the satellite orbit (straight down, known as nadir imagery) are real squares (350x350m), but the further the scanned pixel is from the more distorted it is ([see slide 30 here](https://pages.mtu.edu/~scarn/teaching/GE4250/satellite_lecture_slides)). In order to get the real pixel size we use the `SCAN` and `TRACK` values which reflect the real pixel value (See [VIIRS FAQ](https://earthdata.nasa.gov/faq/firms-faq#ed-viirs-375m-spatial-res)). This is simply done by multiplying `SCAN` and `TRACK` (both in kilometers):

{% highlight python %}
data_gdf['area'] = data_gdf.SCAN * data_gdf.TRACK
data_gdf.drop(['SCAN', 'TRACK'], axis=1, inplace=True)
{% endhighlight %}

## Clipping 
