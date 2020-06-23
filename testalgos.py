# import pandas as pd
# import numpy as np
# from bokeh.io import output_notebook, output_file
# from bokeh.plotting import figure, show, ColumnDataSource
# from bokeh.models.tools import HoverTool
# import math
# from math import pi
# from bokeh.palettes import Category20c
# from bokeh.transform import cumsum
# from bokeh.tile_providers import CARTODBPOSITRON, STAMEN_TERRAIN
# from bokeh.themes import built_in_themes
# from bokeh.io import curdoc
# from bokeh.layouts import gridplot
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.linear_model import LinearRegression
# from sklearn.linear_model import LogisticRegression
#
# from sklearn.model_selection import GridSearchCV
# from bokeh.layouts import gridplot
#
# df_load = pd.read_csv(r"database.csv");
# print(df_load.head())
# df_load = df_load.drop([ 3378, 7512, 20650 ])
# df_load[ 'Year' ] = [ int(x.split('/')[ 2 ]) for x in df_load.iloc[ :, 0 ] ]
# print(df_load.head())
#
# # Create a list of year values
# lst_years = list(df_load[ 'Year' ].unique())
# count_years = [ ]
#
# # Preview years list
# print(lst_years)
#
# # Count the number of records in the dataframe for each year in the lst_years
# for year in lst_years:
#     val = df_load[ df_load[ 'Year' ] == year ]
#     count_years.append(len(val))
#
# # Preview count_years
# print(count_years)
#
# # Build the Earthquakes frequency dataframe using the year and number of earthquakes occuring in each year
#
# df_quake_freq = pd.DataFrame({'Years': lst_years, 'Counts': count_years})
#
# # Preview earthquake freq dataframe
# print(df_quake_freq.head())
#
#
# df_test = pd.read_csv(r"earthquakeTest.csv")
# df_train = df_load.drop(
#             [ 'Depth Error', 'Time', 'Depth Seismic Stations', 'Magnitude Error', 'Magnitude Seismic Stations', 'Azimuthal Gap',
#               'Root Mean Square', 'Source', 'Location Source',
#               'Magnitude Source', 'Status' ], axis=1)
#
#     # Preview of train data
# print(df_train.head())
#
#     # preview df_test
# print(df_test.head())
# print(df_test.columns)
#
# df_test_clean = df_test[ [ 'time', 'latitude', 'longitude', 'mag', 'depth' ] ]
#
# # preview df_test_clean
#
# print(df_test_clean.head())
#
# # Rename fields
#
# df_train = df_train.rename(columns={'Magnitude Type': 'Magnitude_Type'})
# df_test_clean = df_test_clean.rename(
#             columns={'time': 'Date', 'latitude': 'Latitude', 'longitude': 'Longitude', 'mag': 'Magnitude', 'depth': 'Depth'})
#
# # Preview df_train
#
# print(df_train.head())
#
# # Preview df_teset_clean
#
# print(df_test_clean.head())
#
# # Create training and test dataframes
#
# df_testing = df_test_clean[ [ 'Latitude', 'Longitude', 'Magnitude', 'Depth' ] ]
# df_training = df_train[ [ 'Latitude', 'Longitude', 'Magnitude', 'Depth' ] ]
#
# # Preview df_testing
#
# print(df_testing.head())
#
# # Preview df_training
#
# print(df_training.head())
#
# # Remove nulls from our datasets
#
# df_training.dropna()
# df_testing.dropna()
#
# # Create training data features
#
# X = df_training[ [ 'Latitude', 'Longitude' ] ]
# y = df_training[ [ 'Magnitude', 'Depth' ] ]
#
# # Create testing data features
#
# X_new = df_testing[ [ 'Latitude', 'Longitude' ] ]
# y_new = df_testing[ [ 'Magnitude', 'Depth' ] ]
#
# # Import Machine learning libraries
# # from sklearn.model_selection import train_test_split
# # from sklearn.ensemble import RandomForestRegressor
# # from sklearn.model_selection import GridSearchCV
#
# # Use train test split to split our training data into train and test datasets
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # 20% testing 80% training
#
# # Create a model
#
# # model_reg = RandomForestRegressor(random_state=50)
# # model_reg=LinearRegression()
# model_reg=LogisticRegression()
#
# # Train the model
#
# model_reg.fit(X_train, y_train)
#
# # Predict y_test (magnitude and depth) usinf X_test features (LAtitude and Longitude)
#
# results = model_reg.predict(X_test)
#
# # Check the model accuracy score
#
# score = model_reg.score(X_test, y_test) * 100
# print("Accuracy score is")
# print(score)
#
# # Preview predicted earthquakes
#
# print(results)
#
# # Improve the model accuracy by automating hyperparameter tuning
#
# parameters = {'n_estimators': [ 10, 20, 50, 100, 200, 500 ]}
#
# # Create GridsearchCV model
#
# grid_obj = GridSearchCV(model_reg, parameters)
#
# # Train the model
#
# grid_fit = grid_obj.fit(X_train, y_train)
#
# # Select the best fitted model
#
# best_fit = grid_fit.best_estimator_
#
# # Make the prediction
#
# results = best_fit.predict(X_test)
#
# # Check the model accuracy score
#
# score = best_fit.score(X_test, y_test) * 100
# print(score)
#

import pandas as pd
import numpy as np
from bokeh.io import output_notebook, output_file
from bokeh.plotting import figure, show, ColumnDataSource
from bokeh.models.tools import HoverTool
import math
from math import pi
from bokeh.palettes import Category20c
from bokeh.transform import cumsum
from bokeh.tile_providers import CARTODBPOSITRON, STAMEN_TERRAIN
from bokeh.themes import built_in_themes
from bokeh.io import curdoc
from bokeh.layouts import gridplot
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from bokeh.layouts import gridplot

class EarthquakePrediction():
# Load the dataset
    global df_load
    df_load = pd.read_csv(r"database.csv");
    print(df_load.head())

# Create a years field and add it to the dataframe
    df_load = df_load.drop([ 3378, 7512, 20650 ])
    df_load[ 'Year' ] = [ int(x.split('/')[ 2 ]) for x in df_load.iloc[ :, 0 ] ]
    print(df_load.head())

# Create a list of year values
    lst_years = list(df_load[ 'Year' ].unique())
    count_years = [ ]

# Preview years list
    print(lst_years)

# Count the number of records in the dataframe for each year in the lst_years
    for year in lst_years:
        val = df_load[ df_load[ 'Year' ] == year ]
        count_years.append(len(val))

# Preview count_years
    print(count_years)

# Build the Earthquakes frequency dataframe using the year and number of earthquakes occuring in each year

    df_quake_freq = pd.DataFrame({'Years': lst_years, 'Counts': count_years})

# Preview earthquake freq dataframe
    print(df_quake_freq.head())

# Create a ColumnDataSource and a list of year and count values
    source_freq = ColumnDataSource(df_quake_freq)

# Create lists from source_freq ColumnDataSource
    years_list = source_freq.data[ 'Years' ].tolist()
    counts_list = source_freq.data[ 'Counts' ].tolist()

    print(source_freq)


# Define the style of the plots by using a custom style function
    def style(p):
        # Title
        p.title.align = 'center'
        p.title.text_font_size = '20pt'
        p.title.text_font = 'serif'

        # Axis titles
        p.xaxis.axis_label_text_font_size = '14pt'
        p.xaxis.axis_label_text_font_style = 'bold'
        p.yaxis.axis_label_text_font_size = '14pt'
        p.yaxis.axis_label_text_font_style = 'bold'

        #     Tick labels

        p.xaxis.major_label_text_font_size = '12pt'
        p.yaxis.major_label_text_font_size = '12pt'

        # Plot the legend in the top left corner
        p.legend.location = 'top_left'

        return p


# Create the BarChart
    @staticmethod
    def plotBar():
        print("in method")
        # output_notebook()

        # Load the datasource

        cds = ColumnDataSource(data=dict(
            yrs=years_list,
            numQuakes=counts_list

        ))
        print(cds)
        #     Tooltip
        TOOLTIPS = [
            ("Year", "@yrs"),
            ("Number of earthquakes", "@numQuakes")
        ]

        #     Create a figure
        print("hello")
        barChart = figure(title='Frequency of Earthquakes by Year',
                          plot_height=400,
                          plot_width=1000,
                          x_axis_label='Years',
                          y_axis_label='Number of Occurances',
                          x_minor_ticks=2,
                          y_range=(0, df_quake_freq[ 'Counts' ].max() + 100),
                          toolbar_location=None,
                          tooltips=TOOLTIPS,
                          sizing_mode='stretch_both'
                          )
        barChart.vbar(x='yrs', bottom=0, top='numQuakes',
                      color='#009999', width=0.75,
                      legend='Year', source=cds)

    #     Style the barchart

        barChart = EarthquakePrediction.style(barChart)
        print("style")

        # show(barChart)

        return barChart


# EarthquakePrediction.plotBar()


# Create the line chart
    @staticmethod
    def plotLine():
            #     Load the datasource
            cds = ColumnDataSource(data=dict(
                yrs=years_list,
                numQuakes=counts_list
            ))

            TOOLTIPS = [
                ("Year", "@yrs"),
                ("Number of earthquakes", "@numQuakes")
            ]

            p = figure(title='Earthquakes trend by Year',
                       plot_height=400,
                       plot_width=800,
                       x_axis_label='Years',
                       y_axis_label='Number of Occurances',
                       x_minor_ticks=2,
                       y_range=(0, df_quake_freq[ 'Counts' ].max() + 100),
                       toolbar_location=None,
                       tooltips=TOOLTIPS,
                       sizing_mode='stretch_both'
                       )
            p.line(x='yrs', y='numQuakes', color='#009999', line_width=2, legend='Yearly Trend', source=cds)

            p = EarthquakePrediction.style(p)

        # show(p)

            return p


# EarthquakePrediction.plotLine()


# Define the style of the plots by using a custom style function
    def style2(p):
            # Title
            p.title.align = 'center'
            p.title.text_font_size = '20pt'
            p.title.text_font = 'serif'

            # Axis titles
            p.xaxis.axis_label_text_font_size = '14pt'
            p.xaxis.axis_label_text_font_style = 'bold'
            p.yaxis.axis_label_text_font_size = '14pt'
            p.yaxis.axis_label_text_font_size = 'bold'

            #     Tick labels

            p.xaxis.major_label_text_font_size = '12pt'
            p.yaxis.major_label_text_font_size = '12pt'

            # Plot the legend in the top right corner
            p.legend.location = 'top_right'

            return p;

# Create doughnut chart
    @staticmethod
    def plotDoughnut():
            x = dict(df_load[ 'Type' ].value_counts())
            print(x)
            pie_data = pd.Series(x).reset_index(name='value').rename(columns={'index': 'type'})

            #     Preview piedata
            print(pie_data)
            pie_data[ 'angle' ] = pie_data[ 'value' ] / pie_data[ 'value' ].sum() * 2 * pi
            pie_data[ 'color' ] = Category20c[ len(x) ]
            print(pie_data)
            p = figure(title='Types of Earthquakes(1965-2016)',
                       plot_height=400,
                       toolbar_location=None,
                       tools='hover',
                       tooltips="@type: @value",
                       x_range=(-0.5, 1.0), sizing_mode='stretch_both')

            p.annular_wedge(x=0, y=1, inner_radius=0.2, outer_radius=0.35,
                            start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
                            line_color='white', fill_color='color', legend='type', source=pie_data)

            p.axis.axis_label = None
            p.axis.visible = False
            p.grid.grid_line_color = None

            p = EarthquakePrediction.style2(p)

            # show(p)

            return p


    # plotDoughnut()


# Create a Magnitude plot
    @staticmethod
    def plotMagnitude():
            magnitude = [ ]

            # Get the average magnitude value for each year

            for i in df_quake_freq.Years:
                x = df_load[ df_load[ 'Year' ] == i ]
                data_magnitude = sum(x.Magnitude) / len(x.Magnitude)  # Average earthquake magnitude value for each year
                magnitude.append(data_magnitude)

            df_quake_freq[ 'Magnitude' ] = magnitude

            depth = [ ]

            # Get average depth value for each year

            for i in df_quake_freq.Years:
                x = df_load[ df_load[ 'Year' ] == i ]
                data_depth = sum(x.Depth) / len(x.Depth)  # Average earthquake depth value for each year
                depth.append(data_depth)

            df_quake_freq[ 'Depth' ] = depth

            # Get the maximum earthquake magnitude for each year

            max_magnitude = list(df_load.groupby('Year').Magnitude.max())
            df_quake_freq[ 'Max_Magnitude' ] = max_magnitude

            # Preview df_quake_freq
            print(df_quake_freq.head())

            #     Load the data  source
            cds = ColumnDataSource(data=dict(
                yrs=years_list,
                avg_mag=df_quake_freq[ 'Magnitude' ].values.tolist(),
                max_mag=df_quake_freq[ 'Max_Magnitude' ].values.tolist()
            ))

            #     Tooltips
            TOOLTIPS = [
                ("Year", "@yrs"),
                ("Average MAgnitude", "@avg_mag"),
                ("Maximum Magnitude", "@max_mag")
            ]

            #     Create the figure
            mp = figure(title='Maximum and Average Magnitude by Year',
                        plot_width=800,
                        plot_height=500,
                        x_axis_label='Years',
                        y_axis_label='Magnitude',
                        x_minor_ticks=2,
                        y_range=(5, df_quake_freq[ 'Max_Magnitude' ].max() + 1),
                        toolbar_location=None,
                        tooltips=TOOLTIPS,
                        sizing_mode='stretch_both'
                        )

            #     Max Magnitude
            mp.line(x='yrs', y='max_mag', color='#009999', line_width=2, legend='Max Magnitude', source=cds)
            mp.circle(x='yrs', y='max_mag', color='#009999', size=8, fill_color='#009999', source=cds)

            #     Average Magnitude
            mp.line(x='yrs', y='avg_mag', color='orange', line_width=2, legend='Avg Magnitude', source=cds)
            mp.circle(x='yrs', y='max_mag', color='orange', size=8, fill_color='orange', source=cds)

            mp = EarthquakePrediction.style(mp)

            # show(mp)

            return mp


    # plotMagnitude()


# Create Geo Map plot
    @staticmethod
    def plotMap():
            lat = df_load[ 'Latitude' ].values.tolist()
            lon = df_load[ 'Longitude' ].values.tolist()

            lst_lat = [ ]
            lst_lon = [ ]
            i = 0

            #     Convert latitude and longitude values into merc_projection

            for i in range(len(lon)):
                r_major = 6378137.000
                x = r_major * math.radians(lon[ i ])
                scale = x / lon[ i ]
                y = 180.0 / math.pi * math.log(math.tan(math.pi / 4.0 + lat[ i ] * (math.pi / 180.0) / 2.0)) * scale

                lst_lon.append(x)
                lst_lat.append(y)
                i += 1

            df_load[ 'coords_x' ] = lst_lat
            df_load[ 'coords_y' ] = lst_lon

            lats = df_load[ 'coords_x' ].tolist()
            longs = df_load[ 'coords_y' ].tolist()
            mags = df_load[ 'Magnitude' ].tolist()

            # Create datasource

            cds = ColumnDataSource(data=dict(
                lat=lats,
                lon=longs,
                mag=mags
            ))

            # Tooltip

            TOOLTIPS = [
                ("Magnitude", "@mag")
            ]

            # Create figure

            p = figure(title='Earthquake Map',
                       plot_width=1000,
                       plot_height=500,
                       x_range=(-2000000, 6000000),
                       y_range=(-1000000, 7000000),
                       tooltips=TOOLTIPS,
                       sizing_mode='stretch_both'
                       )

            p.circle(x='lon', y='lat', fill_color='#009999', fill_alpha=0.8, source=cds, legend='Quakes 1965-2016')
            p.add_tile(CARTODBPOSITRON)

            # Style the map plot
            # Title

            p.title.align = 'center'
            p.title.text_font_size = '20pt'
            p.title.text_font = 'serif'

            #     Legend
            p.legend.location = 'bottom_right'
            p.legend.background_fill_color = 'black'
            p.legend.background_fill_alpha = 0.8
            # p.legend.click.policy='hide'
            p.legend.label_text_color = 'white'
            p.xaxis.visible = False
            p.yaxis.visible = False
            p.axis.axis_label = None
            p.axis.visible = None
            p.grid.grid_line_color = None

            # show(p)

            return p


    # plotMap()

# Create grid plot

# Make a grid
# grid = gridplot([ [ plotMap(), plotMagnitude(), plotDoughnut() ], [ plotBar(), plotLine() ] ])

# Show the plot

# show(grid)

    df_test = pd.read_csv(r"earthquakeTest.csv")
    df_train = df_load.drop(
            [ 'Depth Error', 'Time', 'Depth Seismic Stations', 'Magnitude Error', 'Magnitude Seismic Stations', 'Azimuthal Gap',
              'Root Mean Square', 'Source', 'Location Source',
              'Magnitude Source', 'Status' ], axis=1)

    # Preview of train data
    print(df_train.head())

    # preview df_test
    print(df_test.head())
    print(df_test.columns)

    df_test_clean = df_test[ [ 'time', 'latitude', 'longitude', 'mag', 'depth' ] ]

# preview df_test_clean

    print(df_test_clean.head())

# Rename fields

    df_train = df_train.rename(columns={'Magnitude Type': 'Magnitude_Type'})
    df_test_clean = df_test_clean.rename(
            columns={'time': 'Date', 'latitude': 'Latitude', 'longitude': 'Longitude', 'mag': 'Magnitude', 'depth': 'Depth'})

# Preview df_train

    print(df_train.head())

# Preview df_teset_clean

    print(df_test_clean.head())

# Create training and test dataframes

    df_testing = df_test_clean[ [ 'Latitude', 'Longitude', 'Magnitude', 'Depth' ] ]
    df_training = df_train[ [ 'Latitude', 'Longitude', 'Magnitude', 'Depth' ] ]

# Preview df_testing

    print(df_testing.head())

# Preview df_training

    print(df_training.head())

# Remove nulls from our datasets

    df_training.dropna()
    df_testing.dropna()

# Create training data features

    X = df_training[ [ 'Latitude', 'Longitude' ] ]
    y = df_training[ [ 'Magnitude', 'Depth' ] ]

# Create testing data features

    X_new = df_testing[ [ 'Latitude', 'Longitude' ] ]
    y_new = df_testing[ [ 'Magnitude', 'Depth' ] ]

# Import Machine learning libraries
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import GridSearchCV

# Use train test split to split our training data into train and test datasets

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # 20% testing 80% training

# Create a model

    model_reg = RandomForestRegressor(random_state=50)

# Train the model

    model_reg.fit(X_train, y_train)

# Predict y_test (magnitude and depth) usinf X_test features (LAtitude and Longitude)

    results = model_reg.predict(X_test)

# Check the model accuracy score

    score = model_reg.score(X_test, y_test) * 100
    print("Accuracy score is")
    print(score)

# Preview predicted earthquakes

    print(results)

# Improve the model accuracy by automating hyperparameter tuning

    parameters = {'n_estimators': [ 10, 20, 50, 100, 200, 500 ]}

# Create GridsearchCV model

    grid_obj = GridSearchCV(model_reg, parameters)

# Train the model

    grid_fit = grid_obj.fit(X_train, y_train)

# Select the best fitted model

    best_fit = grid_fit.best_estimator_

# Make the prediction

    results = best_fit.predict(X_test)

# Check the model accuracy score

    score = best_fit.score(X_test, y_test) * 100
    print(score)

# Predict the earthquakes for the year 2020, and validate the model accuracy using out of sample data

    final_results = best_fit.predict(X_new)

# Check the model accuracy score

    final_score = best_fit.score(X_new, y_new) * 100
    print(final_score)

# Create prediction dataset

# Store prediction results in lists

    lst_Magnitudes = [ ]
    lst_Depth = [ ]
    i = 0

    for r in final_results.tolist():
        lst_Magnitudes.append(final_results[ i ][ 0 ])
        lst_Depth.append(final_results[ i ][ 1 ])
        i += 1

# Create Prediction dataset/dataframe
    global df_results
    df_results = X_new[ [ 'Latitude', 'Longitude' ] ]
    df_results[ 'Magnitude' ] = lst_Magnitudes
    df_results[ 'Depth' ] = lst_Depth
    df_results[ 'Score' ] = final_score
    df_results[ 'Year' ] = 2020

# Preview prediction dataset df_results

    print(df_results.head())



#-------------------------------------------------------------------
# Load the dataset

    df_load = pd.read_csv(r"database.csv");
    df_load = df_load.drop([ 3378, 7512, 20650 ])

# Create a year field and add it to the dataframe

    df_load[ 'Year' ] = [ int(x.split('/')[ 2 ]) for x in df_load.iloc[ :, 0 ] ]

# Create a list of year values
    lst_years = list(df_load[ 'Year' ].unique())
    count_years = [ ]

# Preview years list
    print(lst_years)

# Count the number of records in the dataframe for each year in the lst_years
    for year in lst_years:
        val = df_load[ df_load[ 'Year' ] == year ]
        count_years.append(len(val))

# Preview count_years
    print(count_years)

# Build the Earthquakes frequency dataframe using the year and number of earthquakes occuring in each year
    global df_quake_freq
    df_quake_freq = pd.DataFrame({'Years': lst_years, 'Counts': count_years})

# Preview earthquake freq dataframe
    print(df_quake_freq.head())

# Create a ColumnDataSource and a list of year and count values
    source_freq = ColumnDataSource(df_quake_freq)

# Create lists from source_freq ColumnDataSource
    global years_list
    global counts_list
    years_list = source_freq.data[ 'Years' ].tolist()
    counts_list = source_freq.data[ 'Counts' ].tolist()

    print(source_freq)

# Change the theme to dark theme

    curdoc().theme = 'dark_minimal'


# Create custom style functions to style our plots
# Define the style of our plot using a custom style function

    def style(p):
        # Title
        p.title.align = 'center'
        p.title.text_font_size = '20pt'
        p.title.text_font = 'serif'

        # Axis titles
        p.xaxis.axis_label_text_font_size = '14pt'
        p.xaxis.axis_label_text_font_style = 'bold'
        p.yaxis.axis_label_text_font_size = '14pt'
        p.yaxis.axis_label_text_font_style = 'bold'

        #     Tick labels

        p.xaxis.major_label_text_font_size = '12pt'
        p.yaxis.major_label_text_font_size = '12pt'

        # Plot the legend in the top left corner
        p.legend.location = 'top_left'

        return p


    def style2(p):
        # Title
        p.title.align = 'center'
        p.title.text_font_size = '20pt'
        p.title.text_font = 'serif'

        # Axis titles
        p.xaxis.axis_label_text_font_size = '14pt'
        p.xaxis.axis_label_text_font_style = 'bold'
        p.yaxis.axis_label_text_font_size = '14pt'
        p.yaxis.axis_label_text_font_size = 'bold'

        #     Tick labels

        p.xaxis.major_label_text_font_size = '12pt'
        p.yaxis.major_label_text_font_size = '12pt'

        # Plot the legend in the top right corner
        p.legend.location = 'top_right'

        return p


# Create the bar chart
    @staticmethod
    def plotBar():
        print("in method")
        # output_notebook()

        # Load the datasource

        cds = ColumnDataSource(data=dict(
            yrs=years_list,
            numQuakes=counts_list

        ))
        print(cds)
        #     Tooltip
        TOOLTIPS = [
            ("Year", "@yrs"),
            ("Number of earthquakes", "@numQuakes")
        ]

        #     Create a figure
        print("hello")
        barChart = figure(title='Frequency of Earthquakes by Year',
                          plot_height=400,
                          plot_width=1000,
                          x_axis_label='Years',
                          y_axis_label='Number of Occurances',
                          x_minor_ticks=2,
                          y_range=(0, df_quake_freq[ 'Counts' ].max() + 100),
                          toolbar_location=None,
                          tooltips=TOOLTIPS
                          )
        barChart.vbar(x='yrs', bottom=0, top='numQuakes',
                      color='#009999', width=0.75,
                      legend='Year', source=cds)

        #     Style the barchart

        barChart = EarthquakePrediction.style(barChart)
        print("style")

        # show(barChart)

        return barChart


    # plotBar()


# Create the line chart
    @staticmethod
    def plotLine():
        #     Load the datasource
        cds = ColumnDataSource(data=dict(
            yrs=years_list,
            numQuakes=counts_list
        ))

        TOOLTIPS = [
            ("Year", "@yrs"),
            ("Number of earthquakes", "@numQuakes")
        ]

        p = figure(title='Earthquakes trend by Year',
                   plot_height=400,
                   plot_width=800,
                   x_axis_label='Years',
                   y_axis_label='Number of Occurances',
                   x_minor_ticks=2,
                   y_range=(0, df_quake_freq[ 'Counts' ].max() + 100),
                   toolbar_location=None,
                   tooltips=TOOLTIPS
                   )
        p.line(x='yrs', y='numQuakes', color='#009999', line_width=2, legend='Yearly Trend', source=cds)

        p = EarthquakePrediction.style(p)

    # show(p)

        return p


    # plotLine()


# Create doughnut chart
    @staticmethod
    def plotDoughnut():
        x = dict(df_load[ 'Type' ].value_counts())
        print(x)
        pie_data = pd.Series(x).reset_index(name='value').rename(columns={'index': 'type'})

        #     Preview piedata
        print(pie_data)
        pie_data[ 'angle' ] = pie_data[ 'value' ] / pie_data[ 'value' ].sum() * 2 * pi
        pie_data[ 'color' ] = Category20c[ len(x) ]
        print(pie_data)
        p = figure(title='Types of Earthquakes(1965-2016)',
                   plot_height=400,
                   toolbar_location=None,
                   tools='hover',
                   tooltips="@type: @value",
                   x_range=(-0.5, 1.0))

        p.annular_wedge(x=0, y=1, inner_radius=0.2, outer_radius=0.35,
                        start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
                        line_color='white', fill_color='color', legend='type', source=pie_data)

        p.axis.axis_label = None
        p.axis.visible = False
        p.grid.grid_line_color = None

        p = EarthquakePrediction.style2(p)

        # show(p)

        return p


    # plotDoughnut()


# Create the   Magnitude plot with predictions
    @staticmethod
    def plotMagnitude():
        magnitude = [ ]
        pred_magnitude = [ ]

        # Get the average magnitude value for each year

        for i in df_quake_freq.Years:
            x = df_load[ df_load[ 'Year' ] == i ]
            data_magnitude = sum(x.Magnitude) / len(x.Magnitude)  # Average earthquake magnitude value for each year
            magnitude.append(data_magnitude)
        df_quake_freq[ 'Magnitude' ] = magnitude

        depth = [ ]

        # Get average depth value for each year

        for i in df_quake_freq.Years:
            x = df_load[ df_load[ 'Year' ] == i ]
            data_depth = sum(x.Depth) / len(x.Depth)  # Average earthquake depth value for each year
            depth.append(data_depth)

        df_quake_freq[ 'Depth' ] = depth

    # Get the maximum earthquake magnitude for each year

        max_magnitude = list(df_load.groupby('Year').Magnitude.max())
        df_quake_freq[ 'Max_Magnitude' ] = max_magnitude

        #     Get the average magnitude for the year 2020 from our prediction dataset

        df_results[ 'Mean_Magnitude' ] = df_results[ 'Magnitude' ].mean()
        df_results[ 'Max_Magnitude' ] = df_results[ 'Magnitude' ].max()

        #     Load the datasource

        cds = ColumnDataSource(data=dict(
            yrs=years_list,
            avg_mag=df_quake_freq[ 'Magnitude' ].values.tolist(),
            max_mag=df_quake_freq[ 'Max_Magnitude' ].values.tolist()
        ))

        pred_cds = ColumnDataSource(data=dict(
            yrs=years_list,
            avg_mag=df_results[ 'Mean_Magnitude' ].values.tolist(),
            max_mag=df_results[ 'Max_Magnitude' ].values.tolist()
        ))

        #     Tooltips
        TOOLTIPS = [
            ("Year", "@yrs"),
            ("Average Magnitude", "@avg_mag"),
            ("Maximum Magnitude", "@max_mag")
        ]

        #     Create the figure
        mp = figure(title='Maximum and Average Magnitude by Year',
                    plot_width=800,
                    plot_height=500,
                    x_axis_label='Years',
                    y_axis_label='Magnitude',
                    x_minor_ticks=2,
                    y_range=(5, df_quake_freq[ 'Max_Magnitude' ].max() + 1),
                    toolbar_location=None,
                    tooltips=TOOLTIPS
                    )

        #     Max Magnitude
        mp.line(x='yrs', y='max_mag', color='#009999', line_width=2, legend='Max Magnitude', source=cds)
        mp.circle(x='yrs', y='max_mag', color='#009999', size=8, fill_color='#009999', source=cds)

        #     Average Magnitude
        mp.line(x='yrs', y='avg_mag', color='orange', line_width=2, legend='Avg Magnitude', source=cds)
        mp.circle(x='yrs', y='avg_mag', color='orange', size=8, fill_color='orange', source=cds)

        #    Predicted Max Magnitude
        mp.line(x='yrs', y='max_mag', color='#ccff33', line_width=2, legend='Pred Max/Avg Magnitude', source=pred_cds)
        mp.circle(x='yrs', y='max_mag', color='#ccff33', size=8, fill_color='#ccff33', source=pred_cds)

        #    Predicted Average Magnitude
        mp.line(x='yrs', y='avg_mag', color='#ccff33', line_width=2, legend='Avg Magnitude', source=pred_cds)
        mp.circle(x='yrs', y='avg_mag', color='#ccff33', size=8, fill_color='#ccff33', source=pred_cds)

        mp = EarthquakePrediction.style(mp)

        # show(mp)

        return mp


    # plotMagnitude()


# Create a  Map plot including predictions
    @staticmethod
    def plotMap():
        lat = df_load[ 'Latitude' ].values.tolist()
        lon = df_load[ 'Longitude' ].values.tolist()
        pred_lat = df_results[ 'Latitude' ].values.tolist()
        pred_lon = df_results[ 'Longitude' ].values.tolist()

        lst_lat = [ ]
        lst_lon = [ ]

        lst_pred_lat = [ ]
        lst_pred_lon = [ ]

        i = 0
        j = 0

        #     Convert latitude and longitude values into merc_projection

        for i in range(len(lon)):
            r_major = 6378137.000
            x = r_major * math.radians(lon[ i ])
            scale = x / lon[ i ]
            y = 180.0 / math.pi * math.log(math.tan(math.pi / 4.0 + lat[ i ] * (math.pi / 180.0) / 2.0)) * scale

            lst_lon.append(x)
            lst_lat.append(y)
            i += 1

        #     Convert Predicted latitude and longitude values into merc_projection

        for j in range(len(pred_lon)):
            r_major = 6378137.000
            x = r_major * math.radians(pred_lon[ j ])
            scale = x / pred_lon[ j ]
            y = 180.0 / math.pi * math.log(math.tan(math.pi / 4.0 + pred_lat[ j ] * (math.pi / 180.0) / 2.0)) * scale

            lst_pred_lon.append(x)
            lst_pred_lat.append(y)
            j += 1

        df_load['coords_x'] = lst_lat
        df_load['coords_y'] = lst_lon
        df_results['coords_x'] = lst_pred_lat
        df_results['coords_y'] = lst_pred_lon

        lats = df_load['coords_x'].tolist()
        longs = df_load['coords_y'].tolist()
        mags = df_load['Magnitude'].tolist()
        years = df_load['Year'].tolist()

        pred_lats = df_results['coords_x'].tolist()
        pred_longs = df_results['coords_y'].tolist()
        pred_mags = df_results['Magnitude'].tolist()
        pred_year = df_results['Year'].tolist()

        print(type(pred_mags))
        print(type(mags))
        # Create datasource

        cds = ColumnDataSource(data=dict(
            lat=lats,
            lon=longs,
            mag=mags,
            year=years)
        )

        pred_cds = ColumnDataSource(
            data=dict(
                pred_lat=pred_lats,
                pred_long=pred_longs,
                pred_mag=pred_mags,
                year=pred_year
            ))

        # Tooltip

        TOOLTIPS = [
            ("Magnitude", " @mag"),
            ("Predicted Magnitude", "@pred_mag"),
            ("Year", "@year")
        ]

        # Create figure

        p = figure(title='Earthquake Map',
                   plot_width=1000,
                   plot_height=500,
                   x_range=(-2000000, 6000000),
                   y_range=(-1000000, 7000000),
                   tooltips=TOOLTIPS)

        p.circle(x='lon', y='lat', size='mag', fill_color='#009999', fill_alpha=0.8, source=cds, legend='Quakes 1965-2016')

        # Add circles for our Predicted earthquakes

        p.circle(x='pred_long', y='pred_lat', size='pred_mag', fill_color='#ccff33', fill_alpha=0.8, source=pred_cds,
                 legend='Predicted Quakes 2020')

        p.add_tile(CARTODBPOSITRON)

        # Style the map plot
        # Title

        p.title.align = 'center'
        p.title.text_font_size = '20pt'
        p.title.text_font = 'serif'

        #     Legend
        p.legend.location = 'bottom_right'
        p.legend.background_fill_color = 'black'
        p.legend.background_fill_alpha = 0.8
        # p.legend.click.policy='hide'
        p.legend.label_text_color = 'white'
        p.xaxis.visible = False
        p.yaxis.visible = False
        p.axis.axis_label = None
        p.axis.visible = None
        p.grid.grid_line_color = None

        # show(p)

        return p


    # plotMap()

# Create grid plot

    # from bokeh.layouts import gridplot

# Make a grid
grid = gridplot([ [ EarthquakePrediction.plotMap(), EarthquakePrediction.plotMagnitude(), EarthquakePrediction.plotDoughnut() ], [ EarthquakePrediction.plotBar(), EarthquakePrediction.plotLine() ] ])

# Show the final results

show(grid)

