import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from darts import TimeSeries
from darts.models.forecasting.baselines import NaiveDrift, NaiveSeasonal
from darts.metrics.metrics import mape
from darts.utils.statistics import check_seasonality
from darts.models import ExponentialSmoothing, Theta, RegressionModel
from sklearn.tree import DecisionTreeRegressor


# TODO 2 - 8 -----------------------------------------------------------------------------------------------------------

def ex_2_8():
    passengers_df = pd.read_csv('AirPassengers.csv')
    # print(passengers_df)

    T, val = passengers_df['Month'], passengers_df['#Passengers']

    # plt.plot(T, val)
    # plt.show()

    passengers_ts = TimeSeries.from_dataframe(df=passengers_df, time_col='Month', value_cols='#Passengers')

    # passengers_ts.plot(label='passengers')
    # plt.show()

    train_ts, test_ts = passengers_ts.split_before(0.75)

    # train_ts.plot(label='train')
    # test_ts.plot(label='test')
    # plt.show()

    model_drift = NaiveDrift()
    model_drift.fit(train_ts)
    forecast_drift = model_drift.predict(n=test_ts.n_timesteps)

    # train_ts.plot(label='train')
    # test_ts.plot(label='test')
    # forecast_drift.plot(label='forecast NaiveDrift')
    # plt.show()

    print('MAPE NaiveDrift: ', mape(actual_series=test_ts, pred_series=forecast_drift))

    season, m = check_seasonality(train_ts)
    print('Has seasonality: ', season)
    print('Seasonality period', m)

    model_seasonal = NaiveSeasonal(K=m)
    model_seasonal.fit(train_ts)
    forecast_seasonal = model_seasonal.predict(n=test_ts.n_timesteps)

    # train_ts.plot(label='train')
    # test_ts.plot(label='test')
    # forecast_seasonal.plot(label='forecast NaiveSeasonal')
    # plt.show()

    print('MAPE NaiveSeasonal: ', mape(actual_series=test_ts, pred_series=forecast_seasonal))

    forecast_combined = forecast_drift + forecast_seasonal - train_ts.last_value()

    # train_ts.plot(label='train')
    # test_ts.plot(label='test')
    # forecast_combined.plot(label='forecast combined')
    # plt.show()

    print('MAPE combined: ', mape(actual_series=test_ts, pred_series=forecast_combined))

    model_exp_smooth = ExponentialSmoothing()
    model_exp_smooth.fit(train_ts)
    forecast_exp_smooth = model_exp_smooth.predict(n=test_ts.n_timesteps)

    # train_ts.plot(label='train')
    # test_ts.plot(label='test')
    # forecast_exp_smooth.plot(label='forecast ExponentialSmoothing')
    # plt.show()

    print('MAPE ExponentialSmoothing: ', mape(actual_series=test_ts, pred_series=forecast_exp_smooth))

    model_theta = Theta()
    model_theta.fit(train_ts)
    forecast_theta = model_theta.predict(n=test_ts.n_timesteps)

    # train_ts.plot(label='train')
    # test_ts.plot(label='test')
    # forecast_theta.plot(label='forecast Theta')
    # plt.show()

    print('MAPE Theta: ', mape(actual_series=test_ts, pred_series=forecast_theta))

    # model_dt = RegressionModel(lags=100, model=DecisionTreeRegressor())
    # model_dt.fit(train_ts)
    # forecast_dt = model_dt.predict(n=test_ts.n_timesteps)
    #
    # train_ts.plot(label='train')
    # test_ts.plot(label='test')
    # forecast_dt.plot(label='forecast DecisionTree')
    # plt.show()
    #
    # print('MAPE DecisionTree: ', mape(actual_series=test_ts, pred_series=forecast_dt))


# TODO Resampling ------------------------------------------------------------------------------------------------------

def ex_resampling():
    # generate 40 samples of random data every 4 min
    index = pd.date_range("5/31/2023", periods=40, freq="4T")
    ts = pd.Series(np.random.randint(0, 500, len(index)), index=index)
    print('ts 4min:\n', ts)
    plt.figure()
    ts.plot(title='Original ts')

    # resample to 5 min using sum()
    ts_sum = ts.resample("5T").sum()
    print('ts 5min sum:\n', ts_sum)
    plt.figure()
    ts_sum.plot(title='Resampled ts sum()')

    # resample to 5 min using mean()
    ts_mean = ts.resample("5T").mean()
    print('ts 5min mean:\n', ts_mean)
    plt.figure()
    ts_mean.plot(title='Resampled ts mean()')

    # resample to 5 min using nearest()
    ts_nearest = ts.resample("5T").nearest()
    print('ts 5min nearest:\n', ts_nearest)
    plt.figure()
    ts_nearest.plot(title='Resampled ts nearest()')

    # resample to 5 min using ffill()
    ts_ffill = ts.resample("5T").ffill()
    print('ts 5min ffill:\n', ts_ffill)
    plt.figure()
    ts_ffill.plot(title='Resampled ts ffill()')

    # resample to 5 min using bfill()
    ts_bfill = ts.resample("5T").bfill()
    print('ts 5min bfill:\n', ts_bfill)
    plt.figure()
    ts_bfill.plot(title='Resampled ts bfill()')
    plt.show()


if __name__ == '__main__':
    ex_resampling()

