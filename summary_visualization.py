import csv
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px


covid_airport_csv = pd.read_csv('archive/sorted_covid_impact_on_airport_traffic.csv')


# show mean of PercentOfBaseline each country
def mean_pob_each_country():
    mean_pob = covid_airport_csv.groupby('Country')['PercentOfBaseline'].mean().sort_values(ascending=False).reset_index()
    sns.set(font_scale=1.0)
    plt.figure(figsize=[10, 7])
    sns.barplot(data=mean_pob, x='Country', y='PercentOfBaseline', palette='flare')
    plt.ylabel("Percent of baseline")
    plt.savefig('../mean_pob_each_country.png', format='png')
    plt.show()


# show mean of PercentOfBaseline each city
def mean_pob_each_city():
    mean_pob = covid_airport_csv.groupby('City')['PercentOfBaseline'].mean().sort_values(
        ascending=False).reset_index()
    sns.set(font_scale=1.0)
    plt.figure(figsize=[20, 12])
    sns.barplot(data=mean_pob, x='PercentOfBaseline', y='City', palette='flare')
    plt.xlabel("Percent of baseline")
    plt.savefig('../mean_pob_each_city.png', format='png')
    plt.show()


# show mean of PercentOfBaseline each state
def mean_pob_each_state():
    mean_pob = covid_airport_csv.groupby('State')['PercentOfBaseline'].mean().sort_values(
        ascending=False).reset_index()
    sns.set(font_scale=1.0)
    plt.figure(figsize=[20, 12])
    sns.barplot(data=mean_pob, x='PercentOfBaseline', y='State', palette='flare')
    plt.xlabel("Percent of baseline")
    plt.savefig('../mean_pob_each_state.png', format='png')
    plt.show()


# show mean of PercentOfBaseline each airport
def mean_pob_each_airport():
    mean_pob = covid_airport_csv.groupby('AirportName')['PercentOfBaseline'].mean().sort_values(
        ascending=False).reset_index()
    sns.set(font_scale=1.0)
    plt.figure(figsize=[25, 12])
    sns.barplot(data=mean_pob, x='PercentOfBaseline', y='AirportName', palette='flare')
    plt.xlabel("Percent of baseline")
    plt.ylabel("AirportName")
    plt.savefig('../mean_pob_each_airport.png', format='png')
    plt.show()


# show minimum of PercentOfBaseline each country
def min_pob_each_country():
    min_pob = covid_airport_csv.groupby('Country')['PercentOfBaseline'].min().sort_values(ascending=False).reset_index()
    sns.set(font_scale=1.0)
    plt.figure(figsize=[10, 7])
    sns.barplot(data=min_pob, x='Country', y='PercentOfBaseline', palette='flare')
    plt.ylabel("Percent of baseline")
    plt.savefig('../min_pob_each_country.png', format='png')
    plt.show()


def max_pob_each_country():
    max_pob = covid_airport_csv.groupby('Country')['PercentOfBaseline'].max().sort_values(ascending=False).reset_index()
    sns.set(font_scale=1.0)
    plt.figure(figsize=[10, 7])
    sns.barplot(data=max_pob, x='Country', y='PercentOfBaseline', palette='flare')
    plt.ylabel("Percent of baseline")
    plt.savefig('../max_pob_each_country.png', format='png')
    plt.show()


def min_pob_each_airport():
    min_pob = covid_airport_csv.groupby('AirportName')['PercentOfBaseline'].min().sort_values(ascending=False).reset_index()
    sns.set(font_scale=1.0)
    plt.figure(figsize=[25, 12])
    sns.barplot(data=min_pob, x='PercentOfBaseline', y='AirportName', palette='flare')
    plt.xlabel("Percent of baseline")
    plt.savefig('../min_pob_each_airport.png', format='png')
    plt.show()


def max_pob_each_airport():
    max_pob = covid_airport_csv.groupby('AirportName')['PercentOfBaseline'].max().sort_values(ascending=False).reset_index()
    sns.set(font_scale=1.0)
    plt.figure(figsize=[25, 12])
    sns.barplot(data=max_pob, x='PercentOfBaseline', y='AirportName', palette='flare')
    plt.xlabel("Percent of baseline")
    plt.savefig('../max_pob_each_airport.png', format='png')
    plt.show()


def min_pob_each_city():
    min_pob = covid_airport_csv.groupby('City')['PercentOfBaseline'].min().sort_values(ascending=False).reset_index()
    sns.set(font_scale=1.0)
    plt.figure(figsize=[25, 12])
    sns.barplot(data=min_pob, x='PercentOfBaseline', y='City', palette='flare')
    plt.xlabel("Percent of baseline")
    plt.savefig('../min_pob_each_city.png', format='png')
    plt.show()


def max_pob_each_city():
    max_pob = covid_airport_csv.groupby('City')['PercentOfBaseline'].max().sort_values(ascending=False).reset_index()
    sns.set(font_scale=1.0)
    plt.figure(figsize=[25, 12])
    sns.barplot(data=max_pob, x='PercentOfBaseline', y='City', palette='flare')
    plt.xlabel("Percent of baseline")
    plt.savefig('../max_pob_each_city.png', format='png')
    plt.show()


def min_pob_each_state():
    min_pob = covid_airport_csv.groupby('State')['PercentOfBaseline'].min().sort_values(ascending=False).reset_index()
    sns.set(font_scale=1.0)
    plt.figure(figsize=[25, 12])
    sns.barplot(data=min_pob, x='PercentOfBaseline', y='State', palette='flare')
    plt.xlabel("Percent of baseline")
    plt.savefig('../min_pob_each_state.png', format='png')
    plt.show()


def max_pob_each_state():
    max_pob = covid_airport_csv.groupby('State')['PercentOfBaseline'].max().sort_values(ascending=False).reset_index()
    sns.set(font_scale=1.0)
    plt.figure(figsize=[25, 12])
    sns.barplot(data=max_pob, x='PercentOfBaseline', y='State', palette='flare')
    plt.xlabel("Percent of baseline")
    plt.savefig('../max_pob_each_state.png', format='png')
    plt.show()


# mean_pob_each_country()
# mean_pob_each_state()
# mean_pob_each_city()
# mean_pob_each_airport()
# min_pob_each_country()
# max_pob_each_country()
# min_pob_each_airport()
# max_pob_each_airport()
# min_pob_each_city()
# max_pob_each_city()
# min_pob_each_state()
# max_pob_each_state()

