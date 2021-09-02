import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.tsa.arima_model import ARMA


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


def ch_predict():
    # ADF test
    print('Results of ADF Test:')
    ch_adf_test = adfuller(refined_ch_timeline_csv['PercentOfBaseline'], autolag='AIC')
    ch_adf_output = pd.Series(ch_adf_test[0:4],
                              index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in ch_adf_test[4].items():
        ch_adf_output['Critical Value (%s)' % key] = value
    print(ch_adf_output)

    # KPSS test
    print('Results of KPSS Test:')
    ch_kpss_test = kpss(refined_ch_timeline_csv['PercentOfBaseline'], regression='c')
    ch_kpss_output = pd.Series(ch_kpss_test[0:3], index=['Test Statistic', 'p-value', 'Lags Used'])
    for key, value in ch_kpss_test[3].items():
        ch_kpss_output['Critical Value (%s)' % key] = value
    print(ch_kpss_output)

    # lagged POB
    refined_ch_timeline_csv['diff'] = refined_ch_timeline_csv['PercentOfBaseline'] - refined_ch_timeline_csv['PercentOfBaseline'].shift(1)
    plt.figure(figsize=(25, 12))
    plt.plot(ch_timeline_csv['Date'], refined_ch_timeline_csv['diff'])
    plt.title("Plot for lag of PercentOfBaseline Vs Time for Chile")
    plt.show()
    plt.savefig('../ch_lag_pob_vs_date.png', format='png')

    # lagged POB ADF test
    print('Results of lagged ADF Test:')
    lagged_ch_adf_test = adfuller(refined_ch_timeline_csv['diff'].dropna(), autolag='AIC')
    lagged_ch_adf_output = pd.Series(lagged_ch_adf_test[0:4],
                              index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in lagged_ch_adf_test[4].items():
        lagged_ch_adf_output['Critical Value (%s)' % key] = value
    print(lagged_ch_adf_output)

    # lagged POB KPSS test
    print('Results of KPSS Test:')
    lagged_kpss_test = kpss(refined_ch_timeline_csv['diff'].dropna(), regression='c')
    lagged_kpss_output = pd.Series(lagged_kpss_test[0:3], index=['Test Statistic', 'p-value', 'Lags Used'])
    for key, value in lagged_kpss_test[3].items():
        lagged_kpss_output['Critical Value (%s)' % key] = value
    print(lagged_kpss_output)

    # ARIMA(p,d,q) model, get p d q
    decomposition = seasonal_decompose(x=refined_ch_timeline_csv['PercentOfBaseline'].dropna(), model='multiplicative', period=9)
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    plt.figure(figsize=(25, 12))
    plt.subplot(411)
    plt.plot(refined_ch_timeline_csv['Date'], refined_ch_timeline_csv['PercentOfBaseline'], label='Original')
    plt.legend(loc='best')
    plt.subplot(412)
    plt.plot(refined_ch_timeline_csv['Date'], trend, label='Trend')
    plt.legend(loc='best')
    plt.subplot(413)
    plt.plot(refined_ch_timeline_csv['Date'], seasonal, label='Seasonality')
    plt.legend(loc='best')
    plt.subplot(414)
    plt.plot(refined_ch_timeline_csv['Date'], residual, label='Residuals')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig('../ch_composed_pob_vs_date.png', format='png')
    plt.show()

    # get p=5, q=1
    plot_acf(refined_ch_timeline_csv['diff'].dropna(), zero=False)
    plt.xlim(0, 20)
    plt.xticks(np.arange(0, 20, 1))
    plt.show()
    plot_pacf(refined_ch_timeline_csv['diff'].dropna(), zero=False, lags=40, method='ols', alpha=0.05)
    plt.xticks(np.arange(0, 40, 2))
    plt.show()

    ch_diff = pd.DataFrame(refined_ch_timeline_csv['diff'])
    ch_diff.dropna(inplace=True)
    train = ch_diff.iloc[:212]
    test = ch_diff.iloc[212:]
    model = ARMA(train, order=(5, 1))
    fitted = model.fit()
    print(fitted.summary())
    fc, se, conf = fitted.forecast(25, alpha=0.05)

    # Make as pandas series
    fc_series = pd.Series(fc, index=test.index)
    lower_series = pd.Series(conf[:, 0], index=test.index)
    upper_series = pd.Series(conf[:, 1], index=test.index)

    # Plot
    plt.figure(figsize=(12, 5), dpi=100)
    plt.plot(train, label='training')
    plt.plot(test, label='actual', color='r')
    plt.plot(fc_series, label='forecast', color='g')
    plt.fill_between(lower_series.index, lower_series, upper_series, color='g', alpha=.05)
    plt.title('Forecast vs Actuals')
    plt.legend(loc='best', fontsize=8)
    plt.show()


# mean_pob_vs_date()
# ch_pob_vs_date()
# us_pob_vs_date()
# ca_pob_vs_date()
# au_pob_vs_date()
ch_predict()
