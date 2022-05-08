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
# Loading SHP file:

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

This gives us the final table: (This is its head)
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

The final table will look like:


```python
data_gdf.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ACQ_TIME</th>
      <th>CONFIDENCE</th>
      <th>geometry</th>
      <th>date</th>
      <th>datetime</th>
      <th>area</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0030</td>
      <td>n</td>
      <td>POINT (1029867.816 3239832.490)</td>
      <td>2021-01-01</td>
      <td>2021-01-01 00:30:00</td>
      <td>0.2754</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0030</td>
      <td>n</td>
      <td>POINT (894096.999 3639819.934)</td>
      <td>2021-01-01</td>
      <td>2021-01-01 00:30:00</td>
      <td>0.1995</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0030</td>
      <td>n</td>
      <td>POINT (736967.312 4360360.710)</td>
      <td>2021-01-01</td>
      <td>2021-01-01 00:30:00</td>
      <td>0.3008</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0030</td>
      <td>n</td>
      <td>POINT (666566.639 3717269.176)</td>
      <td>2021-01-01</td>
      <td>2021-01-01 00:30:00</td>
      <td>0.3008</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0030</td>
      <td>n</td>
      <td>POINT (445488.357 3816779.919)</td>
      <td>2021-01-01</td>
      <td>2021-01-01 00:30:00</td>
      <td>0.4402</td>
    </tr>
  </tbody>
</table>
</div>



## Clipping 

In this study, we're only interested in the northern Provinces(Wilayas) of Algeria. This is due to two reasons:

1. We want to study wildfires and those happen almost exclusively in the northern provinces (Some wildfires happen, although they're still much smaller than their northern counterparts).
2. The high temperature of the desert messes up the satellite readings, and causes the hot rooftops to register as fires with nominal confidence. In addition to fires related to oil and gas operations.

We'll also use the Provinces dataset at point. I've used data from [this website](https://data.humdata.org/dataset/cod-ab-dza), make sure to get the shapefile format (SHP) for easy reading. Once downloaded extract all the files (Or only the level 1 files) to a directory of choice, the data is then read the same way as we've read the FIRMS data, while also dropping the unneeded columns and changing the coordinate system:

{% highlight python %}
adm_bnd_gdf = gpd.read_file('dz_admbnd/dza_admbnda_adm1_unhcr_20200120.shp')
adm_bnd_gdf.drop(adm_bnd_gdf.columns.difference(['ADM1_EN', 'geometry']),axis=1,inplace=True)
adm_bnd_gdf = adm_bnd_gdf.to_crs(3857)
{% endhighlight %}

Our map will look like:

{% highlight python %}
fig,ax = plt.subplots(figsize=(8,8))

adm_bnd_gdf.plot(ax=ax)
{% endhighlight %}

![adm_bnd_gdf no clip]({{ "/assets/adm_bnd_no_clip.png" }})

Afterwards, we clip this data to only include provinces north of a certain latitude, we do this by clipping it by creating a rectangle and then clipping it with that rectangle:


{% highlight python %}
poly = Polygon([[-0.5e6,4.5e6],[-0.5e6,4.0e6],[1.2e6,4.0e6],[1.2e6,4.5e6]])
adm_bnd_north_gdf = adm_bnd_gdf.clip(poly)
{% endhighlight %}

{% highlighy python %}
fig,ax = plt.subplots(figsize=(15,15))
adm_bnd_north_gdf.plot(ax=ax)
{% endhighlight %}

![adm_bnd_clip]({{ "/assets/adm_bnd_clip.png" }})

Then clip the rest of the data with the same rectangle (and sorting the values by `datetime`):

```pythong
data_gdf = data_gdf.clip(poly).reset_index().drop('index', axis=1).sort_values(['datetime'])
```

## Time Analysis:

We want to first study the variation of fire surface on the whole region in time. This section should be straight forward normal Pandas operations as we won't need the spatial information.

First, we limit our data to the period of July and August, since the most important fires happened in early August,

```python
july_august_data_gdf = data_gdf.loc[data_gdf.date >= pd.Timestamp('2021-07-01/0000')].loc[data_gdf.date <= pd.Timestamp('2021-08-31/2359')]
```

Then we do a `groupby` on the `date` variable and then sum the `area`s for each day,

```python
area_sums = july_august_data_gdf.groupby('date').area.sum()
```

```python
area_sums
```




    date
    2021-07-01     93.6030
    2021-07-02    114.0144
    2021-07-03     45.4706
    2021-07-04     39.2529
    2021-07-05     45.8119
                    ...   
    2021-08-27     27.7279
    2021-08-28     54.5979
    2021-08-29    101.9258
    2021-08-30    106.9010
    2021-08-31     35.1973
    Name: area, Length: 62, dtype: float64


Afterwards we will simply plot the Series above:

{% highlight python %}
fig,ax = plt.subplots(figsize=(18,12))

x_ticks = [pd.Timestamp(f"2021-{month}-{day}") for month in range(7,9) for day in range(1,31) if pd.Timestamp(f"2021-{month}-{day}").dayofweek == 4]

ax.plot(area_sums.index, area_sums)
ax.set_ylabel(r'$km^2$', fontsize=12)
ax.set_xlabel('date', fontsize=12)
#ax.set_yticks(range(0,180,20), minor=False)
#ax.set_yticks(range(0,183,5), minor=True)
ax.set_xticks(x_ticks, minor=False)
ax.set_xlim(left=pd.Timestamp('2021-07-01'), right=pd.Timestamp('2021-09-01'))
ax.set_xticklabels([f"  Fri.\n {x.month}-{x.day}" for x in x_ticks])
ax.set_title('Active fire surface in northern Algeria during 2021',fontsize=15)
ax.grid(which='both')

plt.savefig('totaltimeanalysis.png')
{% endhighlight %}
![totaltimeanalysis]({{ "/assets/totaltimeanalysis.png" }})

