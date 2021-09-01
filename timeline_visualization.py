import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

timeline_csv = pd.read_csv('archive/sorted_covid_impact_on_airport_traffic.csv')

timeline_csv.columns = ['id', 'AggregationMethod', 'Date', 'Version', 'AirportName',
                        'PercentOfBaseline', 'Centroid', 'City', 'State', 'ISO_3166_2',
                        'Country', 'Geography']

timeline_csv['Date'] = timeline_csv['Date'].astype('datetime64[ns]')

timeline_csv = timeline_csv.drop(columns=['id', 'AggregationMethod', 'Version', 'Centroid', 'ISO_3166_2', 'Geography'])

ch_timeline_csv = timeline_csv[timeline_csv['Country'] == 'Chile']
us_timeline_csv = timeline_csv[timeline_csv['Country'] == 'United States of America (the)']
au_timeline_csv = timeline_csv[timeline_csv['Country'] == 'Australia']
ca_timeline_csv = timeline_csv[timeline_csv['Country'] == 'Canada']

refined_ch_timeline_csv = ch_timeline_csv.drop(columns=['AirportName', 'City', 'State', 'Country'])
refined_us_timeline_csv = us_timeline_csv.drop(columns=['AirportName', 'City', 'State', 'Country'])
refined_au_timeline_csv = au_timeline_csv.drop(columns=['AirportName', 'City', 'State', 'Country'])
refined_ca_timeline_csv = ca_timeline_csv.drop(columns=['AirportName', 'City', 'State', 'Country'])


def mean_pob_vs_date():
    mean_pob = timeline_csv.groupby('Date')['PercentOfBaseline'].mean().sort_values(ascending=False).reset_index()
    plt.figure(figsize=[25, 12])
    plt.plot(mean_pob['PercentOfBaseline'])
    plt.title('Mean percent of baseline vs date')
    plt.savefig('../mean_pob_vs_date.png', format='png')
    plt.show()


def ch_pob_vs_date():
    plt.figure(figsize=[25, 12])
    plt.plot(ch_timeline_csv['Date'], ch_timeline_csv['PercentOfBaseline'])
    plt.title('Percent of baseline vs date for Chile')
    plt.savefig('../ch_mean_pob_vs_date.png', format='png')
    plt.show()


def us_pob_vs_date():
    plt.figure(figsize=[25, 12])
    plt.plot(us_timeline_csv['Date'], us_timeline_csv['PercentOfBaseline'])
    plt.title('Percent of baseline vs date for United States')
    plt.savefig('../us_mean_pob_vs_date.png', format='png')
    plt.show()


def ca_pob_vs_date():
    plt.figure(figsize=[25, 12])
    plt.plot(ca_timeline_csv['Date'], ca_timeline_csv['PercentOfBaseline'])
    plt.title('Percent of baseline vs date for Canada')
    plt.savefig('../ca_mean_pob_vs_date.png', format='png')
    plt.show()


def au_pob_vs_date():
    plt.figure(figsize=[25, 12])
    plt.plot(au_timeline_csv['Date'], au_timeline_csv['PercentOfBaseline'])
    plt.title('Percent of baseline vs date for Australia')
    plt.savefig('../au_mean_pob_vs_date.png', format='png')
    plt.show()


# mean_pob_vs_date()
# ch_pob_vs_date()
# us_pob_vs_date()
# ca_pob_vs_date()
au_pob_vs_date()
