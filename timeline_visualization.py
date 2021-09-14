import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARMA, ARIMA
from datetime import datetime

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

avg_us_timeline_csv = pd.DataFrame(us_timeline_csv.groupby('Date', as_index=False)['PercentOfBaseline'].mean())
avg_ca_timeline_csv = pd.DataFrame(ca_timeline_csv.groupby('Date', as_index=False)['PercentOfBaseline'].mean())

refined_ch_timeline_csv = ch_timeline_csv.drop(columns=['AirportName', 'City', 'State', 'Country'])
refined_us_timeline_csv = us_timeline_csv.drop(columns=['AirportName', 'City', 'State', 'Country'])
refined_au_timeline_csv = au_timeline_csv.drop(columns=['AirportName', 'City', 'State', 'Country'])
refined_ca_timeline_csv = ca_timeline_csv.drop(columns=['AirportName', 'City', 'State', 'Country'])

refined_au_timeline_csv['weekday'] = refined_au_timeline_csv['Date'].dt.dayofweek
refined_ca_timeline_csv['weekday'] = refined_ca_timeline_csv['Date'].dt.dayofweek
refined_us_timeline_csv['weekday'] = refined_us_timeline_csv['Date'].dt.dayofweek
refined_ch_timeline_csv['weekday'] = refined_ch_timeline_csv['Date'].dt.dayofweek

weekday_au_timeline = refined_au_timeline_csv[refined_au_timeline_csv['weekday'] == 6]
weekday_ca_timeline = refined_ca_timeline_csv[refined_ca_timeline_csv['weekday'] == 6]
weekday_us_timeline = refined_us_timeline_csv[refined_us_timeline_csv['weekday'] == 6]
weekday_ch_timeline = refined_ch_timeline_csv[refined_ch_timeline_csv['weekday'] == 6]

sat_au_timeline = refined_au_timeline_csv[refined_au_timeline_csv['weekday'] == 5]
sat_ca_timeline = refined_ca_timeline_csv[refined_ca_timeline_csv['weekday'] == 5]
sat_us_timeline = refined_us_timeline_csv[refined_us_timeline_csv['weekday'] == 5]
sat_ch_timeline = refined_ch_timeline_csv[refined_ch_timeline_csv['weekday'] == 5]

fri_au_timeline = refined_au_timeline_csv[refined_au_timeline_csv['weekday'] == 4]
fri_ca_timeline = refined_ca_timeline_csv[refined_ca_timeline_csv['weekday'] == 4]
fri_us_timeline = refined_us_timeline_csv[refined_us_timeline_csv['weekday'] == 4]
fri_ch_timeline = refined_ch_timeline_csv[refined_ch_timeline_csv['weekday'] == 4]

thu_au_timeline = refined_au_timeline_csv[refined_au_timeline_csv['weekday'] == 3]
thu_ca_timeline = refined_ca_timeline_csv[refined_ca_timeline_csv['weekday'] == 3]
thu_us_timeline = refined_us_timeline_csv[refined_us_timeline_csv['weekday'] == 3]
thu_ch_timeline = refined_ch_timeline_csv[refined_ch_timeline_csv['weekday'] == 3]

wed_au_timeline = refined_au_timeline_csv[refined_au_timeline_csv['weekday'] == 2]
wed_ca_timeline = refined_ca_timeline_csv[refined_ca_timeline_csv['weekday'] == 2]
wed_us_timeline = refined_us_timeline_csv[refined_us_timeline_csv['weekday'] == 2]
wed_ch_timeline = refined_ch_timeline_csv[refined_ch_timeline_csv['weekday'] == 2]

tue_au_timeline = refined_au_timeline_csv[refined_au_timeline_csv['weekday'] == 1]
tue_ca_timeline = refined_ca_timeline_csv[refined_ca_timeline_csv['weekday'] == 1]
tue_us_timeline = refined_us_timeline_csv[refined_us_timeline_csv['weekday'] == 1]
tue_ch_timeline = refined_ch_timeline_csv[refined_ch_timeline_csv['weekday'] == 1]

mon_au_timeline = refined_au_timeline_csv[refined_au_timeline_csv['weekday'] == 0]
mon_ca_timeline = refined_ca_timeline_csv[refined_ca_timeline_csv['weekday'] == 0]
mon_us_timeline = refined_us_timeline_csv[refined_us_timeline_csv['weekday'] == 0]
mon_ch_timeline = refined_ch_timeline_csv[refined_ch_timeline_csv['weekday'] == 0]

sun_avg_us_timeline_csv = pd.DataFrame(
    weekday_us_timeline.groupby('Date', as_index=False)['PercentOfBaseline'].mean())
sun_avg_ca_timeline_csv = pd.DataFrame(
    weekday_ca_timeline.groupby('Date', as_index=False)['PercentOfBaseline'].mean())
fri_avg_us_timeline_csv = pd.DataFrame(
    fri_us_timeline.groupby('Date', as_index=False)['PercentOfBaseline'].mean())
fri_avg_ca_timeline_csv = pd.DataFrame(
    fri_ca_timeline.groupby('Date', as_index=False)['PercentOfBaseline'].mean())
sat_avg_us_timeline_csv = pd.DataFrame(
    sat_us_timeline.groupby('Date', as_index=False)['PercentOfBaseline'].mean())
sat_avg_ca_timeline_csv = pd.DataFrame(
    sat_ca_timeline.groupby('Date', as_index=False)['PercentOfBaseline'].mean())
thu_avg_ca_timeline_csv = pd.DataFrame(
    thu_ca_timeline.groupby('Date', as_index=False)['PercentOfBaseline'].mean())
wed_avg_ca_timeline_csv = pd.DataFrame(
    wed_ca_timeline.groupby('Date', as_index=False)['PercentOfBaseline'].mean())
tue_avg_ca_timeline_csv = pd.DataFrame(
    tue_ca_timeline.groupby('Date', as_index=False)['PercentOfBaseline'].mean())
mon_avg_ca_timeline_csv = pd.DataFrame(
    mon_ca_timeline.groupby('Date', as_index=False)['PercentOfBaseline'].mean())

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
    plt.plot(avg_us_timeline_csv['Date'], avg_us_timeline_csv['PercentOfBaseline'])
    plt.title('Average percent of baseline vs date for United States')
    plt.savefig('../us_mean_pob_vs_date.png', format='png')
    plt.show()


def ca_pob_vs_date():
    plt.figure(figsize=[25, 12])
    plt.plot(avg_ca_timeline_csv['Date'], avg_ca_timeline_csv['PercentOfBaseline'])
    plt.title('Average Percent of baseline vs date for Canada')
    plt.savefig('../ca_mean_pob_vs_date.png', format='png')
    plt.show()


def au_pob_vs_date():
    plt.figure(figsize=[25, 12])
    plt.plot(au_timeline_csv['Date'], au_timeline_csv['PercentOfBaseline'])
    plt.title('Percent of baseline vs date for Australia')
    plt.savefig('../au_mean_pob_vs_date.png', format='png')
    plt.show()


def ca_pob_vs_weekday():
    mean_pob_weekday = refined_ca_timeline_csv.groupby('weekday')['PercentOfBaseline'].mean().sort_values(
        ascending=False).reset_index()
    sns.set(font_scale=1.0)
    plt.figure(figsize=[20, 12])
    sns.barplot(data=mean_pob_weekday, x='weekday', y='PercentOfBaseline', palette='flare')
    plt.xlabel("Percent of baseline")
    plt.savefig('../mean_pob_each_state.png', format='png')
    plt.show()

    plt.figure(figsize=[25, 12])
    plt.plot(sun_avg_ca_timeline_csv['Date'], sun_avg_ca_timeline_csv['PercentOfBaseline'])
    plt.title('Percent of baseline vs date for Canada on Sunday')
    plt.savefig('../ca_mean_pob_sunday.png', format='png')
    plt.show()

    plt.figure(figsize=[25, 12])
    plt.plot(sat_avg_ca_timeline_csv['Date'], sat_avg_ca_timeline_csv['PercentOfBaseline'])
    plt.title('Percent of baseline vs date for Canada on Saturday')
    plt.savefig('../ca_mean_pob_saturday.png', format='png')
    plt.show()

    plt.figure(figsize=[25, 12])
    plt.plot(fri_avg_ca_timeline_csv['Date'], fri_avg_ca_timeline_csv['PercentOfBaseline'])
    plt.title('Percent of baseline vs date for Canada on Friday')
    plt.savefig('../ca_mean_pob_friday.png', format='png')
    plt.show()

    plt.figure(figsize=[25, 12])
    plt.plot(wed_avg_ca_timeline_csv['Date'], wed_avg_ca_timeline_csv['PercentOfBaseline'])
    plt.title('Percent of baseline vs date for Canada on Wednesday')
    plt.savefig('../ca_mean_pob_Wednesday.png', format='png')
    plt.show()

    plt.figure(figsize=[25, 12])
    plt.plot(thu_avg_ca_timeline_csv['Date'], thu_avg_ca_timeline_csv['PercentOfBaseline'])
    plt.title('Percent of baseline vs date for Canada on Thursday')
    plt.savefig('../ca_mean_pob_thursday.png', format='png')
    plt.show()

    plt.figure(figsize=[25, 12])
    plt.plot(tue_avg_ca_timeline_csv['Date'], tue_avg_ca_timeline_csv['PercentOfBaseline'])
    plt.title('Percent of baseline vs date for Canada on Tuesday')
    plt.savefig('../ca_mean_pob_tuesday.png', format='png')
    plt.show()

    plt.figure(figsize=[25, 12])
    plt.plot(mon_avg_ca_timeline_csv['Date'], mon_avg_ca_timeline_csv['PercentOfBaseline'])
    plt.title('Percent of baseline vs date for Canada on Monday')
    plt.savefig('../ca_mean_pob_monday.png', format='png')
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
    refined_ch_timeline_csv['diff'] = refined_ch_timeline_csv['PercentOfBaseline'] - refined_ch_timeline_csv[
        'PercentOfBaseline'].shift(1)
    plt.figure(figsize=(25, 12))
    plt.plot(refined_ch_timeline_csv['Date'], refined_ch_timeline_csv['diff'])
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
    decomposition = seasonal_decompose(x=refined_ch_timeline_csv['PercentOfBaseline'].dropna(), model='multiplicative',
                                       period=9)
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
    ch_diff = ch_diff.set_index(refined_ch_timeline_csv['Date'])
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

    # Prediction
    plt.figure(figsize=(12, 5), dpi=100)
    plt.plot(train, label='training')
    plt.plot(test, label='actual', color='r')
    plt.plot(fc_series, label='forecast', color='g')
    plt.fill_between(lower_series.index, lower_series, upper_series, color='b', alpha=0.1)
    plt.title('Forecast vs Actuals')
    plt.legend(loc='best', fontsize=8)
    plt.savefig('../ch_pob_vs_date_prediction.png', format='png')
    plt.show()


def au_predict():
    # ADF test
    print('Results of ADF Test:')
    au_adf_test = adfuller(refined_au_timeline_csv['PercentOfBaseline'], autolag='AIC')
    au_adf_output = pd.Series(au_adf_test[0:4],
                              index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in au_adf_test[4].items():
        au_adf_output['Critical Value (%s)' % key] = value
    print(au_adf_output)

    # KPSS test
    print('Results of KPSS Test:')
    au_kpss_test = kpss(refined_au_timeline_csv['PercentOfBaseline'], regression='c')
    au_kpss_output = pd.Series(au_kpss_test[0:3], index=['Test Statistic', 'p-value', 'Lags Used'])
    for key, value in au_kpss_test[3].items():
        au_kpss_output['Critical Value (%s)' % key] = value
    print(au_kpss_output)

    # lagged POB
    refined_au_timeline_csv['diff'] = refined_au_timeline_csv['PercentOfBaseline'] - refined_au_timeline_csv[
        'PercentOfBaseline'].shift(1)
    plt.figure(figsize=(25, 12))
    plt.plot(refined_au_timeline_csv['Date'], refined_au_timeline_csv['diff'])
    plt.title("Plot for lag of PercentOfBaseline Vs Time for Australia")
    plt.show()
    plt.savefig('../au_lag_pob_vs_date.png', format='png')

    # lagged POB ADF test
    print('Results of lagged ADF Test:')
    lagged_au_adf_test = adfuller(refined_au_timeline_csv['diff'].dropna(), autolag='AIC')
    lagged_au_adf_output = pd.Series(lagged_au_adf_test[0:4],
                                     index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in lagged_au_adf_test[4].items():
        lagged_au_adf_output['Critical Value (%s)' % key] = value
    print(lagged_au_adf_output)

    # lagged POB KPSS test
    print('Results of KPSS Test:')
    lagged_kpss_test = kpss(refined_au_timeline_csv['diff'].dropna(), regression='c')
    lagged_kpss_output = pd.Series(lagged_kpss_test[0:3], index=['Test Statistic', 'p-value', 'Lags Used'])
    for key, value in lagged_kpss_test[3].items():
        lagged_kpss_output['Critical Value (%s)' % key] = value
    print(lagged_kpss_output)

    # ARIMA(p,d,q) model, get p d q
    decomposition = seasonal_decompose(x=refined_au_timeline_csv['PercentOfBaseline'].dropna(), model='multiplicative',
                                       period=9)
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    plt.figure(figsize=(25, 12))
    plt.subplot(411)
    plt.plot(refined_au_timeline_csv['Date'], refined_au_timeline_csv['PercentOfBaseline'], label='Original')
    plt.legend(loc='best')
    plt.subplot(412)
    plt.plot(refined_au_timeline_csv['Date'], trend, label='Trend')
    plt.legend(loc='best')
    plt.subplot(413)
    plt.plot(refined_au_timeline_csv['Date'], seasonal, label='Seasonality')
    plt.legend(loc='best')
    plt.subplot(414)
    plt.plot(refined_au_timeline_csv['Date'], residual, label='Residuals')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig('../au_composed_pob_vs_date.png', format='png')
    plt.show()

    # get p=5, q=1
    plot_acf(refined_au_timeline_csv['diff'].dropna(), zero=False)
    plt.xlim(0, 20)
    plt.xticks(np.arange(0, 20, 1))
    plt.show()
    plot_pacf(refined_au_timeline_csv['diff'].dropna(), zero=False, lags=40, method='ols', alpha=0.05)
    plt.xticks(np.arange(0, 40, 2))
    plt.show()

    au_diff = pd.DataFrame(refined_au_timeline_csv['diff'])
    au_diff = au_diff.set_index(refined_au_timeline_csv['Date'])
    au_diff.dropna(inplace=True)
    train = au_diff.iloc[:212]
    test = au_diff.iloc[212:]
    model = ARMA(train, order=(7, 1))
    fitted = model.fit()
    print(fitted.summary())
    fc, se, conf = fitted.forecast(44, alpha=0.05)

    # Make as pandas series
    fc_series = pd.Series(fc, index=test.index)
    lower_series = pd.Series(conf[:, 0], index=test.index)
    upper_series = pd.Series(conf[:, 1], index=test.index)

    # Plot
    plt.figure(figsize=(12, 5), dpi=100)
    plt.plot(train, label='training')
    plt.plot(test, label='actual', color='r')
    plt.plot(fc_series, label='forecast', color='g')
    plt.fill_between(lower_series.index, lower_series, upper_series, color='b', alpha=0.1)
    plt.title('Forecast vs Actuals')
    plt.legend(loc='best', fontsize=8)
    plt.savefig('../au_pob_vs_date_prediction.png', format='png')
    plt.show()


def us_predict():
    # ADF test
    print('Results of ADF Test:')
    us_adf_test = adfuller(avg_us_timeline_csv['PercentOfBaseline'], autolag='AIC')
    us_adf_output = pd.Series(us_adf_test[0:4],
                              index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in us_adf_test[4].items():
        us_adf_output['Critical Value (%s)' % key] = value
    print(us_adf_output)

    # KPSS test
    print('Results of KPSS Test:')
    us_kpss_test = kpss(avg_us_timeline_csv['PercentOfBaseline'], regression='c')
    us_kpss_output = pd.Series(us_kpss_test[0:3], index=['Test Statistic', 'p-value', 'Lags Used'])
    for key, value in us_kpss_test[3].items():
        us_kpss_output['Critical Value (%s)' % key] = value
    print(us_kpss_output)

    # lagged POB
    avg_us_timeline_csv['diff'] = avg_us_timeline_csv['PercentOfBaseline'] - avg_us_timeline_csv[
        'PercentOfBaseline'].shift(1)
    plt.figure(figsize=(25, 12))
    plt.plot(avg_us_timeline_csv['Date'], avg_us_timeline_csv['diff'])
    plt.title("Plot for lag of PercentOfBaseline Vs Time for United States")
    plt.show()
    plt.savefig('../us_lag_pob_vs_date.png', format='png')

    # lagged POB ADF test
    print('Results of lagged ADF Test:')
    lagged_us_adf_test = adfuller(avg_us_timeline_csv['diff'].dropna(), autolag='AIC')
    lagged_us_adf_output = pd.Series(lagged_us_adf_test[0:4],
                                     index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in lagged_us_adf_test[4].items():
        lagged_us_adf_output['Critical Value (%s)' % key] = value
    print(lagged_us_adf_output)

    # lagged POB KPSS test
    print('Results of KPSS Test:')
    lagged_kpss_test = kpss(avg_us_timeline_csv['diff'].dropna(), regression='c')
    lagged_kpss_output = pd.Series(lagged_kpss_test[0:3], index=['Test Statistic', 'p-value', 'Lags Used'])
    for key, value in lagged_kpss_test[3].items():
        lagged_kpss_output['Critical Value (%s)' % key] = value
    print(lagged_kpss_output)

    # ARIMA(p,d,q) model, get p d q
    decomposition = seasonal_decompose(x=avg_us_timeline_csv['PercentOfBaseline'].dropna(), model='multiplicative',
                                       period=9)
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    plt.figure(figsize=(25, 12))
    plt.subplot(411)
    plt.plot(avg_us_timeline_csv['Date'], avg_us_timeline_csv['PercentOfBaseline'], label='Original')
    plt.legend(loc='best')
    plt.subplot(412)
    plt.plot(avg_us_timeline_csv['Date'], trend, label='Trend')
    plt.legend(loc='best')
    plt.subplot(413)
    plt.plot(avg_us_timeline_csv['Date'], seasonal, label='Seasonality')
    plt.legend(loc='best')
    plt.subplot(414)
    plt.plot(avg_us_timeline_csv['Date'], residual, label='Residuals')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig('../us_composed_pob_vs_date.png', format='png')
    plt.show()

    # get p=6, q=1
    plot_acf(avg_us_timeline_csv['diff'].dropna(), zero=False)
    plt.xlim(0, 20)
    plt.xticks(np.arange(0, 20, 1))
    plt.show()
    plot_pacf(avg_us_timeline_csv['diff'].dropna(), zero=False, lags=40, method='ols', alpha=0.05)
    plt.xticks(np.arange(0, 40, 2))
    plt.show()

    us_diff = pd.DataFrame(avg_us_timeline_csv['diff'])
    us_diff = us_diff.set_index(avg_us_timeline_csv['Date'])
    us_diff.dropna(inplace=True)
    train = us_diff.iloc[:212]
    test = us_diff.iloc[212:]
    model = ARMA(train, order=(6, 1))
    fitted = model.fit()
    print(fitted.summary())
    fc, se, conf = fitted.forecast(49, alpha=0.05)

    # Make as pandas series
    fc_series = pd.Series(fc, index=test.index)
    lower_series = pd.Series(conf[:, 0], index=test.index)
    upper_series = pd.Series(conf[:, 1], index=test.index)

    # Plot
    plt.figure(figsize=(12, 5), dpi=100)
    plt.plot(train, label='training')
    plt.plot(test, label='actual', color='r')
    plt.plot(fc_series, label='forecast', color='g')
    plt.fill_between(lower_series.index, lower_series, upper_series, color='b', alpha=0.1)
    plt.title('Forecast vs Actuals')
    plt.legend(loc='best', fontsize=8)
    plt.savefig('../us_pob_vs_date_prediction.png', format='png')
    plt.show()


def ca_predict():
    # ADF test
    print('Results of ADF Test:')
    ca_adf_test = adfuller(avg_ca_timeline_csv['PercentOfBaseline'], autolag='AIC')
    ca_adf_output = pd.Series(ca_adf_test[0:4],
                              index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in ca_adf_test[4].items():
        ca_adf_output['Critical Value (%s)' % key] = value
    print(ca_adf_output)

    # KPSS test
    print('Results of KPSS Test:')
    ca_kpss_test = kpss(avg_ca_timeline_csv['PercentOfBaseline'], regression='c')
    ca_kpss_output = pd.Series(ca_kpss_test[0:3], index=['Test Statistic', 'p-value', 'Lags Used'])
    for key, value in ca_kpss_test[3].items():
        ca_kpss_output['Critical Value (%s)' % key] = value
    print(ca_kpss_output)

    # lagged POB
    avg_ca_timeline_csv['diff'] = avg_ca_timeline_csv['PercentOfBaseline'] - avg_ca_timeline_csv[
        'PercentOfBaseline'].shift(1)
    plt.figure(figsize=(25, 12))
    plt.plot(avg_ca_timeline_csv['Date'], avg_ca_timeline_csv['diff'])
    plt.title("Plot for lag of PercentOfBaseline Vs Time for United States")
    plt.show()
    plt.savefig('../ca_lag_pob_vs_date.png', format='png')

    # lagged POB ADF test
    print('Results of lagged ADF Test:')
    lagged_ca_adf_test = adfuller(avg_ca_timeline_csv['diff'].dropna(), autolag='AIC')
    lagged_ca_adf_output = pd.Series(lagged_ca_adf_test[0:4],
                                     index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in lagged_ca_adf_test[4].items():
        lagged_ca_adf_output['Critical Value (%s)' % key] = value
    print(lagged_ca_adf_output)

    # lagged POB KPSS test
    print('Results of KPSS Test:')
    lagged_kpss_test = kpss(avg_ca_timeline_csv['diff'].dropna(), regression='c')
    lagged_kpss_output = pd.Series(lagged_kpss_test[0:3], index=['Test Statistic', 'p-value', 'Lags Used'])
    for key, value in lagged_kpss_test[3].items():
        lagged_kpss_output['Critical Value (%s)' % key] = value
    print(lagged_kpss_output)

    # ARIMA(p,d,q) model, get p d q
    processed_avg_ca_timeline_csv = avg_ca_timeline_csv[True - avg_ca_timeline_csv['PercentOfBaseline'] <= 0]
    decomposition = seasonal_decompose(x=processed_avg_ca_timeline_csv['PercentOfBaseline'].dropna(),
                                       model='multiplicative',
                                       period=9)
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    plt.figure(figsize=(25, 12))
    plt.subplot(411)
    plt.plot(processed_avg_ca_timeline_csv['Date'], processed_avg_ca_timeline_csv['PercentOfBaseline'],
             label='Original')
    plt.legend(loc='best')
    plt.subplot(412)
    plt.plot(processed_avg_ca_timeline_csv['Date'], trend, label='Trend')
    plt.legend(loc='best')
    plt.subplot(413)
    plt.plot(processed_avg_ca_timeline_csv['Date'], seasonal, label='Seasonality')
    plt.legend(loc='best')
    plt.subplot(414)
    plt.plot(processed_avg_ca_timeline_csv['Date'], residual, label='Residuals')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig('../ca_composed_pob_vs_date.png', format='png')
    plt.show()

    # get p=5, q=1
    plot_acf(processed_avg_ca_timeline_csv['diff'].dropna(), zero=False)
    plt.xlim(0, 20)
    plt.xticks(np.arange(0, 20, 1))
    plt.savefig('../ca_acf.png', format='png')
    plt.show()
    plot_pacf(processed_avg_ca_timeline_csv['diff'].dropna(), zero=False, lags=40, method='ols', alpha=0.05)
    plt.xticks(np.arange(0, 40, 2))
    plt.savefig('../ca_pacf.png', format='png')
    plt.show()

    ca_diff = pd.DataFrame(processed_avg_ca_timeline_csv['diff'])
    ca_diff = ca_diff.set_index(processed_avg_ca_timeline_csv['Date'])
    ca_diff.dropna(inplace=True)
    train = ca_diff.iloc[:212]
    test = ca_diff.iloc[212:]

    model = ARIMA(train, order=(6, 1, 2))
    fitted = model.fit(disp=False)
    print(fitted.summary())
    fc, se, conf = fitted.forecast(48, alpha=0.05)

    history_train = list(processed_avg_ca_timeline_csv['diff'][1:212].values)
    history_test = list(processed_avg_ca_timeline_csv['diff'][212:].values)
    rolling_upper = []
    rolling_lower = []
    predictions = list()
    for t in range(len(history_test) - 1):
        print("t:" + str(t))
        new_model = ARIMA(history_train, order=(6, 1, 2))
        new_fitted = new_model.fit(disp=0)
        new_fc, new_se, new_conf = new_fitted.forecast(alpha=0.05)
        predictions.append(new_fc)
        rolling_lower.append(float(new_conf[:, 1]))
        rolling_upper.append(float(new_conf[:, 0]))
        print(rolling_lower)
        obs = history_test[t]
        history_train.append(obs)

    # Make as pandas series
    rolling_series = pd.Series(predictions, index=test.index)
    fc_series = pd.Series(fc, index=test.index)
    lower_series = pd.Series(rolling_lower, index=test.index)
    upper_series = pd.Series(rolling_upper, index=test.index)

    print(lower_series)
    print(upper_series)

    # Plot
    plt.figure(figsize=(12, 5), dpi=100)
    plt.plot(train, label='training')
    plt.plot(test, label='actual', color='r')
    plt.plot(fc_series, label='forecast', color='g')
    plt.fill_between(lower_series.index, lower_series, upper_series, color='b', alpha=0.1)
    plt.title('Forecast vs Actuals')
    plt.legend(loc='best', fontsize=8)
    plt.savefig('../ca_pob_vs_date_prediction.png', format='png')
    plt.show()

    plt.figure(figsize=(12, 5), dpi=100)
    plt.plot(train, label='training')
    plt.plot(test, label='actual', color='r')
    plt.plot(fc_series, label='forecast', color='g')
    plt.plot(rolling_series, label='rolling_forecast', color='y')
    plt.fill_between(lower_series.index, lower_series, upper_series, color='b', alpha=0.1)
    plt.title('Rolling Forecast vs Actuals')
    plt.legend(loc='best', fontsize=8)
    plt.savefig('../ca_pob_vs_date_rolling_prediction.png', format='png')
    plt.show()

    rmse = np.sqrt(((np.array(predictions) - np.array(history_test)) ** 2).mean())
    print(rmse)


# mean_pob_vs_date()
# ch_pob_vs_date()
# us_pob_vs_date()
# ca_pob_vs_date()
# au_pob_vs_date()
# ch_predict()
# us_predict()
# ca_predict()
# au_predict()
ca_pob_vs_weekday()
