
# Analysing and Visualizing the trends in Global Warming temperature rise based on greenhouse gases like CO2, N2O and CH4 and Predicting the 2 deg Celsius temperature rise.

## **Introduction**

Global warming is one of the most pressing environmental issues of our time, driven primarily by the increase in greenhouse gases such as carbon dioxide (CO2), methane (CH4), and nitrous oxide (N2O). These gases trap heat in the Earth's atmosphere, leading to a rise in global temperatures, which in turn causes significant climatic changes. According to the Intergovernmental Panel on Climate Change (IPCC), the global average temperature has already risen by approximately 1.1°C above pre-industrial levels. Understanding the trends in temperature rise and predicting future temperature scenarios is crucial for devising effective mitigation and adaptation strategies.

This project aims to analyze and visualize the trends in global warming by focusing on the contributions of CO2, CH4, and N2O emissions. By leveraging advanced machine learning techniques, the project will predict when the global temperature is likely to breach the critical 2°C increase threshold above pre-industrial levels. This prediction is essential for policymakers and researchers to understand the urgency of reducing greenhouse gas emissions and to formulate strategies to combat climate change effectively.

The analysis will involve collecting historical data on greenhouse gas concentrations and global temperature changes. For instance, atmospheric CO2 levels have risen from 280 parts per million (ppm) in the pre-industrial era to over 410 ppm in recent years. Similarly, methane and nitrous oxide levels have increased significantly due to human activities such as agriculture, fossil fuel extraction, and industrial processes. The project will apply data visualization techniques to uncover patterns and trends and use predictive modeling to forecast future temperature rises. This comprehensive approach will provide valuable insights into the relationship between greenhouse gas emissions and global warming, highlighting the need for immediate and sustained action to protect our planet.


# Data

## 1) Dataset

for this project we have used following Datasets

[1] Annual greenhouse gas emissions by activity and by region.
        source: https://climatedata.imf.org/pages/access-data

[2] Hannah Ritchie, Pablo Rosado and Max Roser (2023) - “CO₂ and Greenhouse Gas Emissions” Published online at OurWorldInData.org. Retrieved from: 'https://ourworldindata.org/co2-and-greenhouse-gas-emissions' [Online Resource]

## 2) Data Preprocessing

#### importing required packages
```bash
    import pandas as pd
    import numpy as np
```


#### Dataset1 : GlobalLandTemperaturesByCity
```bash
    df = pd.read_csv('GlobalLandTemperaturesByCity.csv')
    df.info()
```
#### Exploratory Data Analysis
```bash
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 8599212 entries, 0 to 8599211
    Data columns (total 7 columns):
    #   Column                         Dtype  
    ---  ------                         -----  
    0   dt                             object 
    1   AverageTemperature             float64
    2   AverageTemperatureUncertainty  float64
    3   City                           object 
    4   Country                        object 
    5   Latitude                       object 
    6   Longitude                      object 
    dtypes: float64(2), object(5)
    memory usage: 459.2+ MB
```
```bash
    df.isna().sum()
```
```bash
    dt                                    0
    AverageTemperature               364130
    AverageTemperatureUncertainty    364130
    City                                  0
    Country                               0
    Latitude                              0
    Longitude                             0
    dtype: int64
```
```bash
    df.describe()
```
```bash
    	AverageTemperature	AverageTemperatureUncertainty
    count	8.235082e+06	8.235082e+06
    mean	1.672743e+01	1.028575e+00
    std	    1.035344e+01	1.129733e+00
    min	    -4.270400e+01	3.400000e-02
    25%	    1.029900e+01	3.370000e-01
    50%	    1.883100e+01	5.910000e-01
    75%	    2.521000e+01	1.349000e+00
    max	    3.965100e+01	1.539600e+01

```
``` bash
    df.columns
```
```bash
    Index(['dt', 'AverageTemperature', 'AverageTemperatureUncertainty', 'City',
       'Country', 'Latitude', 'Longitude'],
      dtype='object')
```

```bash
    df.head()
```
```bash
    	dt	AverageTemperature	AverageTemperatureUncertainty	City	Country	Latitude	Longitude
0	1743-11-01	    6.068	    1.737	    Århus	Denmark	57.05N	10.33E
1	1743-12-01	    NaN	        NaN	        Århus	Denmark	57.05N	10.33E
2	1744-01-01	    NaN	        NaN	        Århus	Denmark	57.05N	10.33E
3	1744-02-01	    NaN	        NaN	        Århus	Denmark	57.05N	10.33E
4	1744-03-01	    NaN	        NaN	        Århus	Denmark	57.05N	10.33E
```
```bash
    df.tail()
```
```bash
	    dt	AverageTemperature	AverageTemperatureUncertainty	City	Country	Latitude	Longitude
8599207	2013-05-01	11.464	0.236	Zwolle	Netherlands	52.24N	5.26E
8599208	2013-06-01	15.043	0.261	Zwolle	Netherlands	52.24N	5.26E
8599209	2013-07-01	18.775	0.193	Zwolle	Netherlands	52.24N	5.26E
8599210	2013-08-01	18.025	0.298	Zwolle	Netherlands	52.24N	5.26E
8599211	2013-09-01	NaN	NaN	Zwolle	Netherlands	52.24N	5.26E
```
#### Convert the 'dt' column to datetime format
```bash
    # Convert the 'dt' column to datetime format
    df['dt'] = pd.to_datetime(df['dt'])

    # Extract the year from the 'dt' column
    df['Year'] = df['dt'].dt.year

    # Calculate the annual average temperature for each city
    city_annual_avg_temp = df.groupby(['City', 'Country', 'Year'])['AverageTemperature'].mean().reset_index()

    # Calculate the annual average temperature for each country
    country_annual_avg_temp = city_annual_avg_temp.groupby(['Country', 'Year'])['AverageTemperature'].mean().reset_index()

    # Save the results to an Excel file
    filename = 'country_annual_average_temperatures.xlsx'
    country_annual_avg_temp.to_excel(filename, index=False)

    print(f"Country-level annual average temperatures have been saved to '{filename}'")
```
    Country-level annual average temperatures have been saved to 'country_annual_average_temperatures.xlsx'

```bash
    df_4 = pd.read_excel('country_annual_average_temperatures.xlsx')
    df_4.head()
```
```bash
	Country	    Year	AverageTemperature
0	Afghanistan	1833	13.091150
1	Afghanistan	1834	13.093600
2	Afghanistan	1835	13.959233
3	Afghanistan	1836	NaN
4	Afghanistan	1837	13.186690
```
```bash
    df_4.info()
```
```bash
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 32881 entries, 0 to 32880
    Data columns (total 3 columns):
    #   Column              Non-Null Count  Dtype  
    ---  ------              --------------  -----  
    0   Country             32881 non-null  object 
    1   Year                32881 non-null  int64  
    2   AverageTemperature  31556 non-null  float64
    dtypes: float64(1), int64(1), object(1)
    memory usage: 770.8+ KB
```
```bash
    df_4.isna().sum()
```
        Country                  0
        Year                     0
        AverageTemperature    1325
        dtype: int64

```bash
    # Extract unique values from a specific column
    # Replace 'column_name' with the actual column name from which you want unique values
    unique_values = df['Country'].unique()

    # Print the unique values
    print(unique_values)
    print(len(unique_values))
```
    ['Denmark' 'Turkey' 'Kazakhstan' 'China' 'Spain' 'Germany' 'Nigeria'
    'Iran' 'Russia' 'Canada' "Côte D'Ivoire" 'United Kingdom' 'Saudi Arabia'
    'Japan' 'United States' 'India' 'Benin' 'United Arab Emirates' 'Mexico'
    'Venezuela' 'Ghana' 'Ethiopia' 'Australia' 'Yemen' 'Indonesia' 'Morocco'
    'Pakistan' 'France' 'Libya' 'Burma' 'Brazil' 'South Africa' 'Syria'
    'Egypt' 'Algeria' 'Netherlands' 'Malaysia' 'Portugal' 'Ecuador' 'Italy'
    'Uzbekistan' 'Philippines' 'Madagascar' 'Chile' 'Belgium' 'El Salvador'
    'Romania' 'Peru' 'Colombia' 'Tanzania' 'Tunisia' 'Turkmenistan' 'Israel'
    'Eritrea' 'Paraguay' 'Greece' 'New Zealand' 'Vietnam' 'Cameroon' 'Iraq'
    'Afghanistan' 'Argentina' 'Azerbaijan' 'Moldova' 'Mali'
    'Congo (Democratic Republic Of The)' 'Thailand'
    'Central African Republic' 'Bosnia And Herzegovina' 'Bangladesh'
    'Switzerland' 'Equatorial Guinea' 'Cuba' 'Lebanon' 'Mozambique' 'Serbia'
    'Angola' 'Somalia' 'Norway' 'Nepal' 'Poland' 'Ukraine' 'Guinea Bissau'
    'Malawi' 'Burkina Faso' 'Slovakia' 'Congo' 'Belarus' 'Gambia'
    'Czech Republic' 'Hungary' 'Burundi' 'Zimbabwe' 'Bulgaria' 'Haiti'
    'Puerto Rico' 'Sri Lanka' 'Nicaragua' 'Zambia' 'Honduras' 'Taiwan'
    'Bolivia' 'Guinea' 'Ireland' 'Senegal' 'Latvia' 'Qatar' 'Albania'
    'Tajikistan' 'Kenya' 'Guatemala' 'Finland' 'Sierra Leone' 'Sweden'
    'Botswana' 'Guyana' 'Austria' 'Uganda' 'Armenia' 'Dominican Republic'
    'Jordan' 'Djibouti' 'Sudan' 'Lithuania' 'Rwanda' 'Jamaica' 'Togo'
    'Macedonia' 'Cyprus' 'Gabon' 'Slovenia' 'Bahrain' 'Swaziland' 'Niger'
    'Lesotho' 'Liberia' 'Uruguay' 'Chad' 'Bahamas' 'Mauritania' 'Panama'
    'Suriname' 'Cambodia' 'Montenegro' 'Mauritius' 'Papua New Guinea'
    'Iceland' 'Croatia' 'Reunion' 'Oman' 'Costa Rica' 'South Korea'
    'Hong Kong' 'Singapore' 'Estonia' 'Georgia' 'Mongolia' 'Laos' 'Namibia']
    159
    
#### Filling NA values with moving average

```bash
    # Load the dataset
    temperature_data = pd.read_excel('country_annual_average_temperatures.xlsx')

    # Sort the data by country and year
    temperature_data = temperature_data.sort_values(by=['Country', 'Year'])

    # Function to fill missing temperatures
    def fill_missing_temperatures(df):
        countries = df['Country'].unique()
        filled_data = pd.DataFrame()

        for country in countries:
            country_data = df[df['Country'] == country].copy()
            country_data['AverageTemperature'] = country_data['AverageTemperature'].interpolate(method='linear', limit_direction='forward')

            for i in range(len(country_data)):
                if pd.isna(country_data.iloc[i]['AverageTemperature']):
                    start_idx = max(0, i-5)
                    end_idx = i
                    past_5_years = country_data.iloc[start_idx:end_idx]['AverageTemperature']
                    if past_5_years.count() > 0:
                        country_data.at[country_data.index[i], 'AverageTemperature'] = past_5_years.mean()

            filled_data = pd.concat([filled_data, country_data])

        return filled_data

    # Fill missing temperatures
    filled_temperature_data = fill_missing_temperatures(temperature_data)

    # Save the filled dataset to a new file
    filled_temperature_data.to_csv('filled_temperature_dataset.csv', index=False)
```
```bash
    df_tmp = pd.read_csv('filled_temperature_dataset.csv')
    df_tmp.isna().sum()
```
    Country               0
    Year                  0
    AverageTemperature    0
    dtype: int64

#### Dataset2 : ghg-emissions-by-gas.csv

#### Load the required packages
```bash
    import pandas as pd
    import numpy as np
```
#### Exploratory Data Analysis
```bash
    # Load your DataFrame
    df2 = pd.read_csv('ghg-emissions-by-gas.csv')
    df2.head()
```

    	Entity	Code	Year	Annual nitrous oxide emissions in CO₂ equivalents	Annual methane emissions in CO₂ equivalents	Annual CO₂ emissions
    0	Afghanistan	AFG	1850	223008.40	3594926.5	3520884.0
    1	Afghanistan	AFG	1851	227659.61	3615134.5	3561188.2
    2	Afghanistan	AFG	1852	232190.92	3635346.8	3596619.0
    3	Afghanistan	AFG	1853	236528.19	3655563.5	3630340.0
    4	Afghanistan	AFG	1854	240597.22	3675785.0	3662827.5
```bash
    df2.info()
```
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 41520 entries, 0 to 41519
    Data columns (total 6 columns):
    #   Column                                             Non-Null Count  Dtype  
    ---  ------                                             --------------  -----  
    0   Entity                                             41520 non-null  object 
    1   Code                                               37541 non-null  object 
    2   Year                                               41520 non-null  int64  
    3   Annual nitrous oxide emissions in CO₂ equivalents  38060 non-null  float64
    4   Annual methane emissions in CO₂ equivalents        37195 non-null  float64
    5   Annual CO₂ emissions                               41520 non-null  float64
    dtypes: float64(3), int64(1), object(2)
    memory usage: 1.9+ MB

```bash
    df2.isna().sum()
```
    Entity                                                  0
    Code                                                 3979
    Year                                                    0
    Annual nitrous oxide emissions in CO₂ equivalents    3460
    Annual methane emissions in CO₂ equivalents          4325
    Annual CO₂ emissions                                    0
    dtype: int64

```bash
    unique_values = df['Entity'].unique()
    print(unique_values)
    len(unique_values)
```
    ['Afghanistan' 'Africa' 'Albania' 'Algeria' 'Andorra' 'Angola' 'Anguilla'
    'Antarctica' 'Antigua and Barbuda' 'Argentina' 'Armenia' 'Aruba' 'Asia'
    'Asia (excl. China and India)' 'Australia' 'Austria' 'Azerbaijan'
    'Bahamas' 'Bahrain' 'Bangladesh' 'Barbados' 'Belarus' 'Belgium' 'Belize'
    'Benin' 'Bermuda' 'Bhutan' 'Bolivia' 'Bonaire Sint Eustatius and Saba'
    'Bosnia and Herzegovina' 'Botswana' 'Brazil' 'British Virgin Islands'
    'Brunei' 'Bulgaria' 'Burkina Faso' 'Burundi' 'Cambodia' 'Cameroon'
    'Canada' 'Cape Verde' 'Central African Republic' 'Chad' 'Chile' 'China'
    'Christmas Island' 'Colombia' 'Comoros' 'Congo' 'Cook Islands'
    'Costa Rica' "Cote d'Ivoire" 'Croatia' 'Cuba' 'Curacao' 'Cyprus'
    'Czechia' 'Democratic Republic of Congo' 'Denmark' 'Djibouti' 'Dominica'
    'Dominican Republic' 'East Timor' 'Ecuador' 'Egypt' 'El Salvador'
    'Equatorial Guinea' 'Eritrea' 'Estonia' 'Eswatini' 'Ethiopia' 'Europe'
    'Europe (excl. EU-27)' 'Europe (excl. EU-28)' 'European Union (27)'
    'European Union (28)' 'Faroe Islands' 'Fiji' 'Finland' 'France'
    'French Polynesia' 'Gabon' 'Gambia' 'Georgia' 'Germany' 'Ghana' 'Greece'
    'Greenland' 'Grenada' 'Guatemala' 'Guinea' 'Guinea-Bissau' 'Guyana'
    'Haiti' 'High-income countries' 'Honduras' 'Hong Kong' 'Hungary'
    'Iceland' 'India' 'Indonesia' 'Iran' 'Iraq' 'Ireland' 'Israel' 'Italy'
    'Jamaica' 'Japan' 'Jordan' 'Kazakhstan' 'Kenya' 'Kiribati' 'Kosovo'
    'Kuwait' 'Kuwaiti Oil Fires' 'Kyrgyzstan' 'Laos' 'Latvia'
    'Least developed countries (Jones et al.)' 'Lebanon' 'Leeward Islands'
    'Lesotho' 'Liberia' 'Libya' 'Liechtenstein' 'Lithuania'
    'Low-income countries' 'Lower-middle-income countries' 'Luxembourg'
    'Macao' 'Madagascar' 'Malawi' 'Malaysia' 'Maldives' 'Mali' 'Malta'
    'Marshall Islands' 'Mauritania' 'Mauritius' 'Mexico'
    'Micronesia (country)' 'Moldova' 'Mongolia' 'Montenegro' 'Montserrat'
    'Morocco' 'Mozambique' 'Myanmar' 'Namibia' 'Nauru' 'Nepal' 'Netherlands'
    'New Caledonia' 'New Zealand' 'Nicaragua' 'Niger' 'Nigeria' 'Niue'
    'North America' 'North America (excl. USA)' 'North Korea'
    'North Macedonia' 'Norway' 'OECD (Jones et al.)' 'Oceania' 'Oman'
    'Pakistan' 'Palau' 'Palestine' 'Panama' 'Panama Canal Zone'
    'Papua New Guinea' 'Paraguay' 'Peru' 'Philippines' 'Poland' 'Portugal'
    'Qatar' 'Romania' 'Russia' 'Rwanda' 'Ryukyu Islands' 'Saint Helena'
    'Saint Kitts and Nevis' 'Saint Lucia' 'Saint Pierre and Miquelon'
    'Saint Vincent and the Grenadines' 'Samoa' 'Sao Tome and Principe'
    'Saudi Arabia' 'Senegal' 'Serbia' 'Seychelles' 'Sierra Leone' 'Singapore'
    'Sint Maarten (Dutch part)' 'Slovakia' 'Slovenia' 'Solomon Islands'
    'Somalia' 'South Africa' 'South America' 'South Korea' 'South Sudan'
    'Spain' 'Sri Lanka' 'St. Kitts-Nevis-Anguilla' 'Sudan' 'Suriname'
    'Sweden' 'Switzerland' 'Syria' 'Taiwan' 'Tajikistan' 'Tanzania'
    'Thailand' 'Togo' 'Tonga' 'Trinidad and Tobago' 'Tunisia' 'Turkey'
    'Turkmenistan' 'Turks and Caicos Islands' 'Tuvalu' 'Uganda' 'Ukraine'
    'United Arab Emirates' 'United Kingdom' 'United States'
    'Upper-middle-income countries' 'Uruguay' 'Uzbekistan' 'Vanuatu'
    'Venezuela' 'Vietnam' 'Wallis and Futuna' 'World' 'Yemen' 'Zambia'
    'Zimbabwe']
    240
```bash
    # Rename the column 'Entity' to 'Country'
    df2.rename(columns={'Entity': 'Country'}, inplace=True)

    # Save the DataFrame to a new CSV file
    df2.to_csv('ghg-emissions-by-gas.csv', index=False)

    print("DataFrame saved to ghg-emissions-by-gas.csv")
```
    DataFrame saved to ghg-emissions-by-gas.csv
```bash
    df_ghg = pd.read_csv('ghg-emissions-by-gas.csv')
    df_ghg.head()
```
```bash
    	Country	Code	Year	Annual nitrous oxide emissions in CO₂ equivalents	Annual methane emissions in CO₂ equivalents	Annual CO₂ emissions
    0	Afghanistan	AFG	1850	223008.40	3594926.5	3520884.0
    1	Afghanistan	AFG	1851	227659.61	3615134.5	3561188.2
    2	Afghanistan	AFG	1852	232190.92	3635346.8	3596619.0
    3	Afghanistan	AFG	1853	236528.19	3655563.5	3630340.0
    4	Afghanistan	AFG	1854	240597.22	3675785.0	3662827.5
```

#### Merging the two Datasets
```bash
    # Load the first dataset
    temperature_data = pd.read_csv('filled_temperature_dataset.csv')

    # Load the second dataset
    emissions_data = pd.read_csv('ghg-emissions-by-gas.csv')

    # Merge the datasets on 'country' and 'year'
    merged_data = pd.merge(temperature_data, emissions_data, on=['Country', 'Year'], how='inner')

    # Display the first few rows of the merged dataset
    merged_data.head()
```
	    Country	Year	AverageTemperature	Code	Annual nitrous oxide emissions in CO₂ equivalents	Annual methane emissions in CO₂ equivalents	Annual CO₂ emissions
    0	Afghanistan	1850	13.185427	AFG	223008.40	3594926.5	3520884.0
    1	Afghanistan	1851	13.391073	AFG	227659.61	3615134.5	3561188.2
    2	Afghanistan	1852	13.337948	AFG	232190.92	3635346.8	3596619.0
    3	Afghanistan	1853	13.270833	AFG	236528.19	3655563.5	3630340.0
    4	Afghanistan	1854	13.481042	AFG	240597.22	3675785.0	3662827.5

#### creating a csv file for merged dataset
```bash
    # Save the merged dataset to a new CSV file (optional)
    merged_data.to_csv('merged_data_inner.csv', index=False)
```
Thus 'merged_data_inner.csv' is the final dataset after applying EDA, cleaning and merging datasets. 

Now this dataset can be used for model building.

#### EDA on merged_data_inner.csv
```bash
    df = pd.read_csv('merged_data_inner.csv')
    df.head()
```
	    Country	Year	AverageTemperature	Code	Annual nitrous oxide emissions in CO₂ equivalents	Annual methane emissions in CO₂ equivalents	Annual CO₂ emissions
    0	Afghanistan	1850	13.185427	AFG	223008.40	3594926.5	3520884.0
    1	Afghanistan	1851	13.391073	AFG	227659.61	3615134.5	3561188.2
    2	Afghanistan	1852	13.337948	AFG	232190.92	3635346.8	3596619.0
    3	Afghanistan	1853	13.270833	AFG	236528.19	3655563.5	3630340.0
    4	Afghanistan	1854	13.481042	AFG	240597.22	3675785.0	3662827.5
```bash
    df.info()
```
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 24248 entries, 0 to 24247
    Data columns (total 7 columns):
    #   Column                                             Non-Null Count  Dtype  
    ---  ------                                             --------------  -----  
    0   Country                                            24248 non-null  object 
    1   Year                                               24248 non-null  int64  
    2   AverageTemperature                                 24248 non-null  float64
    3   Code                                               24248 non-null  object 
    4   Annual nitrous oxide emissions in CO₂ equivalents  24248 non-null  float64
    5   Annual methane emissions in CO₂ equivalents        24248 non-null  float64
    6   Annual CO₂ emissions                               24248 non-null  float64
    dtypes: float64(4), int64(1), object(2)
    memory usage: 1.3+ MB

```bash
    df.isna().sum()
```
    Country                                              0
    Year                                                 0
    AverageTemperature                                   0
    Code                                                 0
    Annual nitrous oxide emissions in CO₂ equivalents    0
    Annual methane emissions in CO₂ equivalents          0
    Annual CO₂ emissions                                 0
    dtype: int64
```bash

```
## Model building


``` bash
# importing required packages

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.pyplot as plt

```
```bash

# Load your dataset
data = pd.read_csv('merged_data_inner.csv')

# Preprocessing: Handle missing values if any
data = data.dropna()

# Feature selection
features = ['Year', 'Annual nitrous oxide emissions in CO₂ equivalents',
            'Annual methane emissions in CO₂ equivalents', 'Annual CO₂ emissions']
X = data[features]
y = data['AverageTemperature']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
lin_model = LinearRegression()
lin_model.fit(X_train, y_train)

rf_model = RandomForestRegressor(n_estimators=2000, random_state=42)
rf_model.fit(X_train, y_train)

xgb_model = XGBRegressor(n_estimators=2000, learning_rate=0.01, max_depth=4, random_state=42)
xgb_model.fit(X_train, y_train)


# Model evaluation
lin_y_pred = lin_model.predict(X_test)
rf_y_pred = rf_model.predict(X_test)
xgb_y_pred = xgb_model.predict(X_test)

lin_mse = mean_squared_error(y_test, lin_y_pred)
rf_mse = mean_squared_error(y_test, rf_y_pred)
xgb_mse = mean_squared_error(y_test, xgb_y_pred)

lin_rmse = np.sqrt(lin_mse)
rf_rmse = np.sqrt(rf_mse)
xgb_rmse = np.sqrt(xgb_mse)

lin_r2 = r2_score(y_test, lin_y_pred)
rf_r2 = r2_score(y_test, rf_y_pred)
xgb_r2 = r2_score(y_test, xgb_y_pred)

print(f'Linear Regression Mean Squared Error: {lin_mse}')
print(f'Random Forest Mean Squared Error: {rf_mse}')
print(f'XGBoost Mean Squared Error: {xgb_mse}')

print(f'Linear Regression Root Mean Squared Error: {lin_rmse}')
print(f'Random Forest Root Mean Squared Error: {rf_rmse}')
print(f'XGBoost Root Mean Squared Error: {xgb_rmse}')

print(f'Linear Regression R-squared: {lin_r2}')
print(f'Random Forest R-squared: {rf_r2}')
print(f'XGBoost R-squared: {xgb_r2}')

# Predict future temperatures
future_years = np.arange(2014, 2101).reshape(-1, 1)
future_features = np.hstack([future_years, np.zeros((future_years.shape[0], 3))])  # Assuming emissions remain constant

# Set hypothetical growth rates for emissions (2% per year for illustration)
growth_rate = 0.02
base_no2 = data['Annual nitrous oxide emissions in CO₂ equivalents'].mean()
base_ch4 = data['Annual methane emissions in CO₂ equivalents'].mean()
base_co2 = data['Annual CO₂ emissions'].mean()

for i in range(future_years.shape[0]):
    year = 2014 + i
    future_features[i, 1] = base_no2 * (1 + growth_rate) ** (year - 2014)  # NO2
    future_features[i, 2] = base_ch4 * (1 + growth_rate) ** (year - 2014)  # CH4
    future_features[i, 3] = base_co2 * (1 + growth_rate) ** (year - 2014)  # CO2

# Predictions
lin_future_temperatures = lin_model.predict(future_features)
rf_future_temperatures = rf_model.predict(future_features)
xgb_future_temperatures = xgb_model.predict(future_features)

# Find the year when temperature breaches 2 degrees
pre_industrial_temp = data[data['Year'] < 1900]['AverageTemperature'].mean()

lin_breach_year = future_years[np.argmax(lin_future_temperatures >= pre_industrial_temp + 2)][0]
rf_breach_year = future_years[np.argmax(rf_future_temperatures >= pre_industrial_temp + 2)][0]
xgb_breach_year = future_years[np.argmax(xgb_future_temperatures >= pre_industrial_temp + 2)][0]

# Output results
print(f'Future Years: {future_years.flatten()}')
print(f'Linear Regression future temperatures: {lin_future_temperatures}')
print(f'Random Forest future temperatures: {rf_future_temperatures}')
print(f'XGBoost future temperatures: {xgb_future_temperatures}')
print(f'Linear Regression: The global temperature is expected to breach the 2°C mark in the year {lin_breach_year}')
print(f'Random Forest: The global temperature is expected to breach the 2°C mark in the year {rf_breach_year}')
print(f'XGBoost: The global temperature is expected to breach the 2°C mark in the year {xgb_breach_year}')

```
Linear Regression Mean Squared Error: 56.71443921640465\
Random Forest Mean Squared Error: 7.142229155316167\
XGBoost Mean Squared Error: 26.27684427416718\
Linear Regression Root Mean Squared Error: 7.530898964692373\
Random Forest Root Mean Squared Error: 2.672494930830771\
XGBoost Root Mean Squared Error: 5.12609444647357\
Linear Regression R-squared: 0.017506308283300687\
Random Forest R-squared: 0.8762714542037912\
XGBoost R-squared: 0.5447925767354884

Future Years: [2014 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025 2026 2027
 2028 2029 2030 2031 2032 2033 2034 2035 2036 2037 2038 2039 2040 2041
 2042 2043 2044 2045 2046 2047 2048 2049 2050 2051 2052 2053 2054 2055
 2056 2057 2058 2059 2060 2061 2062 2063 2064 2065 2066 2067 2068 2069
 2070 2071 2072 2073 2074 2075 2076 2077 2078 2079 2080 2081 2082 2083
 2084 2085 2086 2087 2088 2089 2090 2091 2092 2093 2094 2095 2096 2097
 2098 2099 2100]

Linear Regression future temperatures: [19.46588923 19.47593111 19.48589603 19.49578243 19.50558875 19.51531339
 19.52495472 19.53451105 19.54398071 19.55336195 19.562653   19.57185207
 19.58095731 19.58996684 19.59887876 19.6076911  19.61640188 19.62500906
 19.63351058 19.64190432 19.65018813 19.6583598  19.6664171  19.67435773
 19.68217937 19.68987963 19.69745608 19.70490625 19.71222762 19.71941761
 19.72647358 19.73339287 19.74017273 19.74681038 19.75330297 19.7596476
 19.76584132 19.7718811  19.77776387 19.78348648 19.78904574 19.79443837
 19.79966105 19.80471037 19.80958286 19.814275   19.81878317 19.82310369
 19.82723281 19.83116671 19.83490147 19.83843312 19.8417576  19.84487075
 19.84776836 19.85044611 19.8528996  19.85512436 19.8571158  19.85886926
 19.86037998 19.8616431  19.86265368 19.86340666 19.86389689 19.86411911
 19.86406797 19.863738   19.86312361 19.86221913 19.86101875 19.85951656
 19.85770651 19.85558245 19.85313809 19.85036705 19.84726277 19.84381859
 19.84002773 19.83588323 19.83137804 19.82650493 19.82125655 19.8156254
 19.80960381 19.80318398 19.79635794]

Random Forest future temperatures: [16.94672521 16.91794767 16.73163132 16.62007997 16.03314746 15.02346347
 15.04948209 15.53549164 15.64142855 15.66974635 15.72221528 15.5031836
 16.0788469  15.74296614 15.68134663 14.99852169 14.64582514 13.73805231
 13.84901104 13.90671095 13.03824065 13.14810803 13.60758967 13.87357868
 13.53277056 14.27229984 14.42849559 14.38941907 14.87633213 16.19643511
 15.82274294 16.8744909  16.9984774  17.55451117 17.43819954 18.64800267
 19.8742922  20.21863946 22.89731293 22.47395349 22.80650069 22.52320304
 22.56044881 22.47849845 22.60647467 22.92371894 22.65296258 22.99575585
 23.56289866 23.5168199  22.64376148 22.00914285 22.06726041 21.75300401
 17.06736121 16.35059569 14.45574084 14.23401582 13.64737075 13.94965738
 13.71417511 13.24723944 13.20321502 13.07505256 13.10558083 13.25194884
 13.59657592 14.08828291 14.07347515 12.47599919 12.5093273  12.74611979
 12.88076441 13.47478195 15.58060848 16.10232668 16.92494465 18.59983902
 20.08401091 20.39455594 19.51777403 18.17411845 17.45412101 17.00730402
 16.91409806 15.86513125 16.22594759]

XGBoost future temperatures: [16.743631 16.60596  15.820957 16.40665  16.40665  16.369524 16.369524
 15.651032 15.651032 15.651032 15.384628 15.384628 15.384628 15.384628
 15.711363 15.06634  15.06634  14.994061 14.994061 15.565962 15.565962
 17.969093 18.426743 18.426743 18.426743 18.653305 18.653305 18.918028
 18.918028 19.014723 20.401197 20.401197 19.221699 19.388433 19.388433
 19.388433 19.297592 19.297592 19.297592 19.493767 19.493767 20.283686
 20.283686 20.53175  20.49596  20.49596  20.49596  20.49596  20.655342
 20.655342 20.655342 20.776093 20.776093 20.476648 21.329231 21.329231
 21.329231 20.73654  17.496063 17.496063 17.317184 17.45781  17.45781
 15.427316 15.427316 15.427316 15.837697 15.837697 15.837697 16.223524
 15.041973 15.041973 15.118814 15.118814 15.118814 15.118814 16.191677
 16.191677 16.191677 16.536499 16.41026  16.41026  16.41026  16.41026
 16.716711 16.716711 16.399773]

Linear Regression: The global temperature is expected to breach the 2°C mark in the year 2064\
Random Forest: The global temperature is expected to breach the 2°C mark in the year 2050\
XGBoost: The global temperature is expected to breach the 2°C mark in the year 2044