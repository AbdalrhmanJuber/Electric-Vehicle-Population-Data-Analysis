# Electric Vehicle Population Data Analysis

## Project Overview

This project performs comprehensive data preprocessing and exploratory data analysis (EDA) on the "Electric Vehicle Population Data" dataset provided by the State of Washington. The dataset includes information about registered Battery Electric Vehicles (BEVs) and Plug-in Hybrid Electric Vehicles (PHEVs) in Washington state.

##Dataset Information

- The dataset consists of 210,165 records and 17 features, capturing details such as:

- Vehicle Identification Number (VIN)

- County and city of registration

- Make and model

- Electric vehicle type

- Electric range

- Model year

- Location (latitude, longitude)

- Dataset Source

## Data Processing Steps

1. Data Cleaning

- Identified and handled missing values (dropping records and mean/median imputation).

2. Feature Engineering

- Applied One-hot Encoding to categorical features (Make, Model).

- Normalized numerical features (Min-Max Scaling) on Electric Range.

3. Exploratory Data Analysis (EDA)

- Calculated descriptive statistics (mean, median, standard deviation).

- Analyzed spatial distribution using geographic visualization.

- Investigated popularity trends for various vehicle makes and models.

- Examined correlations between numerical features.

## Visualizations


- Histograms (model year, electric range).

- Scatter plots and boxplots analyzing relationships between Model Year and Electric Range.

- Bar charts depicting distribution across cities and counties.

## Python Libraries Used

- Pandas

- NumPy

- Matplotlib

- Seaborn

- GeoPandas

- Scikit-learn (for data preprocessing)

## How to Run the Project

Clone this repository.

Install required libraries:

pip install pandas numpy matplotlib seaborn geopandas scikit-learn contextily

Run the provided Python script:

python Final_Code.py

## Contributors

- Ali Shaikh Qasem (ID: 1212171)

- Abdalrahman Juber (ID: 1211769)

## Supervisors

- Dr. Ismail Khater

- Dr. Yazan Abu Farha

## University

Birzeit University, Faculty of Engineering and Technology, Department of Electrical and Computer Engineering.

## Date

October 30, 2024
