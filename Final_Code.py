import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import contextily as ctx
import seaborn as sns
from shapely.geometry import Point



if __name__ == '__main__':

    df = pd.read_csv("Electric_Vehicle_Population_Data.csv")
    '''
  
    ### Document Missing Values ###
    print(df.shape) # calculating number of row x column
    print(df.info()) # count the number of non null values for all
    
    missing_values_count = df.isnull().sum()
    
    print(missing_values_count) # to find the missing data for each column
    print(df.isnull().any(axis=1).sum()) # to find the number of missing data (number of rows)
    print(100 * df.isnull().any(axis=1).sum() / df.shape[0], '%') # to find the percentage of missing data
     '''

    '''
   ### Missing Value Strategies ###

    ## Handling Missing Data Through Dropping ##
    print(f"Number of empty records = {df.isnull().all(axis=1).sum()}") # w
    df[df.isnull().all(axis=1)] # to drop all records that are null

    df_cleaned_drop = df.dropna() # removes all rows that contain at least one missing value (NaN) from the DataFrame
     '''


    '''
    # Mean/Median Imputation:
    numeric_cols = df.select_dtypes(include=['number']) #select only numeric columns from the DataFrame.
    df_mean_imputed = df.fillna(numeric_cols.mean())

    df_median_imputed = df.fillna(numeric_cols.median())

    print("Original Data Summary:\n", df.describe())
    print("Dropped Rows Summary:\n", df_cleaned_drop.describe())
    print("Mean Imputed Summary:\n", df_mean_imputed.describe())
    print("Median Imputed Summary:\n", df_median_imputed.describe())
    
    '''
    '''
    ### Feature Encoding ###


    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    # Fit the encoder on the 'Make' and 'Model' columns and transform them
    encoded_data = encoder.fit_transform(df[['Make', 'Model']])

    # Get the feature names for the new one-hot encoded columns
    encoded_columns = encoder.get_feature_names_out(['Make', 'Model'])

    # Create a DataFrame with the one-hot encoded data and the new column names
    df_encoded = pd.DataFrame(encoded_data, columns=encoded_columns)

    # Concatenate the new encoded columns with the original DataFrame (dropping the original 'Make' and 'Model' columns)
    df_final = pd.concat([df.drop(columns=['Make', 'Model']), df_encoded], axis=1)

    print(df_encoded.head())
    
    '''
    '''

    #### Normalization ###

    # Select Electric Range
    numerical_cols = ['Electric Range']

    # Create a MinMaxScaler instance
    scaler = MinMaxScaler()

    # Fit and transform the numerical columns
    df_normalized = df.copy()  # Create a copy of the DataFrame
    df_normalized[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    # Display the first few rows of the normalized DataFrame
    print(df_normalized['Electric Range'].head())
    print(df['Electric Range'].head())
    
    '''

    ##################### point 5: Discriptive statistics (median, mode, standard deviation) ###########################
    # select the numeric values from the dataset
    # numeric_df = df.select_dtypes(include='number')
    # std_values = numeric_df.std()
    # mean_values = numeric_df.mean()
    # median_values = numeric_df.median()
    # print("Mean values are: \n", mean_values, "\n")
    # print("Median values are: \n", median_values, "\n")
    # print("Stabdard deviation values are: \n", std_values,  "\n")

    ###################### point 6: spatial distribution ########################

    #
    # # Clean and extract coordinates from 'Vehicle Location'
    # df['Vehicle Location'] = df['Vehicle Location'].str.replace(r'POINT \(', '', regex=True).str.replace(r'\)', '',
    #                                                                                                      regex=True)
    # df[['Longitude', 'Latitude']] = df['Vehicle Location'].str.split(expand=True)
    # df['Longitude'] = pd.to_numeric(df['Longitude'])
    # df['Latitude'] = pd.to_numeric(df['Latitude'])
    #
    # # Drop NaNs
    # df = df.dropna(subset=['Longitude', 'Latitude'])
    #
    # # Convert DataFrame to GeoDataFrame
    # gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['Longitude'], df['Latitude']))
    #
    # # ** Set the CRS explicitly to EPSG:4326 (WGS84) **
    # gdf.set_crs(epsg=4326, inplace=True)
    #
    # # Separate BEV and PHEV for different colors
    # bev = gdf[gdf['Electric Vehicle Type'] == 'Battery Electric Vehicle (BEV)']
    # phev = gdf[gdf['Electric Vehicle Type'] == 'Plug-in Hybrid Electric Vehicle (PHEV)']
    #
    # # Plotting the map
    # fig, ax = plt.subplots(figsize=(10, 10))
    #
    # # Plot BEV locations in blue
    # bev.plot(ax=ax, color='blue', markersize=5, label='Battery Electric Vehicle (BEV)', alpha=0.6)
    #
    # # Plot PHEV locations in red
    # phev.plot(ax=ax, color='red', markersize=5, label='Plug-in Hybrid Electric Vehicle (PHEV)', alpha=0.6)
    #
    # # Adding the basemap (contextily OpenStreetMap)
    # ctx.add_basemap(ax, crs=gdf.crs, source=ctx.providers.OpenStreetMap.Mapnik)
    #
    # # Add title, labels, and legend
    # ax.set_title('Electric Vehicles (BEV and PHEV)', fontsize=15)
    # ax.set_xlabel('Longitude', fontsize=12)
    # ax.set_ylabel('Latitude', fontsize=12)
    # plt.legend()
    #
    #
    # # Show the map
    # plt.show()

    ################ point 7: model popularity ###################

    # Filter the DataFrame to include only the "BEVs"
    bev_df = df[df['Electric Vehicle Type'] == "Battery Electric Vehicle (BEV)"]

    # Filter the DataFrame to include only the "PHEV"
    phev_df = df[df['Electric Vehicle Type'] == "Plug-in Hybrid Electric Vehicle (PHEV)"]
    #
    # ### make (car type) ###
    # # Count the number of records (samples) for each make for BEVs
    # bev_make_counts = bev_df['Make'].value_counts()
    #
    # # Count the number of records (samples) for each make for PHEVs
    # phev_make_counts = phev_df['Make'].value_counts()
    #
    # # Plot for BEVs
    # plt.figure(figsize=(12, 6))
    # plt.subplot(1, 2, 1)
    # bev_make_counts[: 30].plot(kind='bar', color='blue')
    # plt.title('Make Popularity for BEVs')
    # plt.xlabel('Make')
    # plt.ylabel('Number of records')
    #
    # # Plot for PHEVs
    # plt.subplot(1, 2, 2)
    # phev_make_counts[: 30].plot(kind='bar', color='green')
    # plt.title('Make Popularity for PHEVs')
    # plt.xlabel('Make')
    # plt.ylabel('Number of records')
    #
    # # Show plots
    # plt.tight_layout()
    # plt.show()

    ####################################
    #### model ####

    # # Count the number of records (samples) for each make for BEVs
    # bev_model_counts = bev_df['Model'].value_counts()
    #
    # # Count the number of records (samples) for each make for PHEVs
    # phev_model_counts = phev_df['Model'].value_counts()
    #
    # # Plot for BEVs
    # plt.figure(figsize=(12, 6))
    # plt.subplot(1, 2, 1)
    # bev_model_counts[: 30].plot(kind='bar', color='blue')
    # plt.title('Model Popularity for BEVs')
    # plt.xlabel('Model')
    # plt.ylabel('Number of records')
    #
    # # Plot for PHEVs
    # plt.subplot(1, 2, 2)
    # phev_model_counts[: 30].plot(kind='bar', color='green')
    # plt.title('Model Popularity for PHEVs')
    # plt.xlabel('Model')
    # plt.ylabel('Number of records')
    #
    # # Show plots
    # plt.tight_layout()
    # plt.show()

    ################### point 8: relationship between every pair of numeric features #################
    # # select the numeric values from the dataset
    # numeric_df = df.select_dtypes(include='number')
    # corr_matrix = numeric_df.corr() # find the correlation matrix
    # print("Correlations matrix: \n", corr_matrix )
     
    ############################ point 9: Data exploration visualizations ###################################
    ###### histogram for model year ############
    # unique_years = sorted(df['Model Year'].unique())
    # plt.figure(figsize=(10, 6))
    # df['Model Year'].hist(bins=len(unique_years), edgecolor='black', rwidth=0.8)
    # plt.title('Model Year Distribution')
    # plt.xlabel('Model Year')
    # plt.ylabel('Frequency')
    # plt.xticks(unique_years, rotation=45)
    # plt.show()
    
    ########## histogram for electric range ###############
    # bins = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]

    # plt.figure(figsize=(10, 6))
    # df['Electric Range'].hist(bins=bins, edgecolor='black', rwidth=0.8)

    # plt.title('Electric Range Distribution')
    # plt.xlabel('Electric Range (miles)')
    # plt.ylabel('Frequency')
    # plt.xticks(bins)  
    # plt.show()

    ######## scatter plot ( model year aganinst electric range) #################
    # plt.figure(figsize=(10, 6))
    # sns.scatterplot(data=df, x='Model Year', y='Electric Range', alpha=0.7)
    # plt.title('Electric Range vs. Model Year')
    # plt.xlabel('Model Year')
    # plt.ylabel('Electric Range')
    # plt.show()

    ############### box plot between model year and electric range ###################
    # plt.figure(figsize=(10, 6))
    # sns.boxplot(data=df, x='Model Year', y='Electric Range', palette='Set2')
    # plt.title('Electric Range by Model Year')
    # plt.xlabel('Vehicle Type')
    # plt.ylabel('Electric Range (miles)')
    # plt.show()

    
    ######################## point 10: Comparative Visualization #######################
    ## County  ###
    # # Count the number of records (samples) for each county for BEVs
    # bev_county_counts = bev_df['County'].value_counts()
    #
    # # Count the number of records (samples) for each county for PHEVs
    # phev_county_counts = phev_df['County'].value_counts()
    #
    # # Plot for BEVs
    # plt.figure(figsize=(12, 6))
    # plt.subplot(1, 2, 1)
    # bev_county_counts[: 30].plot(kind='bar', color='blue')
    # plt.title('County Popularity for BEVs')
    # plt.xlabel('County')
    # plt.ylabel('Number of records')
    #
    # # Plot for PHEVs
    # plt.subplot(1, 2, 2)
    # phev_county_counts[: 30].plot(kind='bar', color='green')
    # plt.title('County Popularity for PHEVs')
    # plt.xlabel('County')
    # plt.ylabel('Number of records')
    #
    # # Show plots
    # plt.tight_layout()
    # plt.show()
    # ######################################
    # ## City  ###
    # # Count the number of records (samples) for each city for BEVs
    # bev_city_counts = bev_df['City'].value_counts()
    #
    # # Count the number of records (samples) for each city for PHEVs
    # phev_city_counts = phev_df['City'].value_counts()
    #
    # # Plot for BEVs
    # plt.figure(figsize=(12, 6))
    # plt.subplot(1, 2, 1)
    # bev_city_counts[: 30].plot(kind='bar', color='blue')
    # plt.title('City Popularity for BEVs')
    # plt.xlabel('City')
    # plt.ylabel('Number of records')
    #
    # # Plot for PHEVs
    # plt.subplot(1, 2, 2)
    # phev_city_counts[: 30].plot(kind='bar', color='green')
    # plt.title('City Popularity for PHEVs')
    # plt.xlabel('City')
    # plt.ylabel('Number of records')
    #
    # # Show plots
    # plt.tight_layout()
    # plt.show()