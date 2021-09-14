import csv
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

covid_airport_csv = pd.read_csv('archive/covid_impact_on_airport_traffic.csv')


# drop null value in data
def drop_null():
    covid_airport_csv.dropna()


# give some basic information
def overall_information():
    # total number of lines
    number_of_lines = covid_airport_csv.shape[0]
    number_of_columns = covid_airport_csv.shape[1]
    print("number_of_lines:"+str(number_of_lines))
    print("number_of_columns:" + str(number_of_columns))


    # unique value count
    airport_names = covid_airport_csv['AirportName'].unique()
    country_names = covid_airport_csv['Country'].unique()
    ISO_names = covid_airport_csv['ISO_3166_2'].unique()
    state_names = covid_airport_csv['State'].unique()
    city_names = covid_airport_csv['City'].unique()
    print(airport_names)
    print('airport_numbers:' + str(len(airport_names)))
    print(country_names)
    print('country_numbers:' + str(len(country_names)))
    print(ISO_names)
    print('ISO_numbers:' + str(len(ISO_names)))
    print(state_names)
    print('state_numbers:' + str(len(state_names)))
    print(city_names)
    print('city_numbers:' + str(len(city_names)))

    # number of airports in each country
    airport_number_each_country = covid_airport_csv.groupby('Country')['AirportName'].nunique().reset_index()
    airport_name_each_country = covid_airport_csv.groupby('Country')['AirportName'].unique().reset_index()
    au_airport = airport_name_each_country['AirportName'].loc[0].tolist()
    ca_airport = airport_name_each_country['AirportName'].loc[1].tolist()
    ch_airport = airport_name_each_country['AirportName'].loc[2].tolist()
    us_airport = airport_name_each_country['AirportName'].loc[3].tolist()
    print(airport_number_each_country)
    sns.set(font_scale=1.0)
    plt.figure(figsize=[25, 12])
    sns.barplot(data=airport_number_each_country, x='Country', y='AirportName', palette='flare')
    plt.ylabel("AirportNumber")
    plt.savefig('../airport_number_each_country.png', format='png')
    plt.show()
    print(airport_name_each_country)

    # number of days each airport
    days_each_airport = covid_airport_csv.groupby('AirportName')['Date'].apply(len).reset_index()
    print(days_each_airport)


# sort record by date each airport
def sort_by_date():
    sorted_by_date_each_airport = covid_airport_csv.groupby('AirportName').apply(lambda x: x.sort_values('Date'))
    sorted_by_date_each_airport.to_csv('archive/sorted_covid_impact_on_airport_traffic.csv')
    temp = pd.read_csv('archive/sorted_covid_impact_on_airport_traffic.csv')
    temp.columns = ['AN', 'id', 'AggregationMethod', 'Date', 'Version',
                    'AirportName', 'PercentOfBaseline', 'Centroid', 'City', 'State',
                    'ISO_3166_2', 'Country', 'Geography']
    temp.drop('AN', axis=1, inplace=True)
    temp.drop('id', axis=1, inplace=True)
    temp.reset_index()
    temp.drop_duplicates()
    temp.to_csv('archive/sorted_covid_impact_on_airport_traffic.csv')


drop_null()
sort_by_date()
overall_information()

