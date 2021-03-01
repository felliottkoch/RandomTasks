from sklearn.kernel_ridge import KernelRidge
from src.analysis_tools import *
from pathlib import Path

import pandas as pd
import numpy as np
import datetime
import os


def _calc_time_int(ts: datetime.datetime, points_per_day=96):
    points_per_hour = points_per_day / 24
    if points_per_hour == 1:
        return ts.hour
    elif 1 < points_per_hour <= 60:
        return ts.hour * 60 + ts.minute
    elif 60 < points_per_hour <= 3600:
        return ts.hour * 2600 + ts.minute * 60 + ts.second
    else:
        raise ValueError('point_per_day has inappropriate value {}'.format(points_per_day))


def _parse_data(data_raw: pd.Series, lag: int=None, points_per_day=96):
    x = []
    y = []
    if lag is None:
        lag = 0
    for index in range(lag, len(data_raw)):
        time_int = _calc_time_int(data_raw.index[index], points_per_day=points_per_day)
        ts = data_raw.index[index]
        x.append(np.array([data_raw.iloc[indx] for indx in range(index-lag, index)] + [time_int, ts.weekday(), ]))
        y.append(data_raw.iloc[index])
    return np.array(x), np.array(y)


def _parse_data_for_prediction(data_raw: pd.Series, lag: int=None, points_per_day=96):
    if lag is None:
        lag = 0
    if len(data_raw) < lag:
        raise ValueError('data do not span lag')
    x = [data_raw[indx] for indx in range(len(data_raw)-lag, len(data_raw))]
    ts = data_raw.index[-1]
    time_int = _calc_time_int(data_raw.index[-1] + datetime.timedelta(hours=24/points_per_day))
    return np.array([np.array(x + [time_int, ts.weekday()])])


def _parse_data_from_df(data_raw: pd.DataFrame, endpoint=DataFrameKeys.esb_const_avail, lag=None):
    return _parse_data(data_raw[endpoint], lag=lag)


class ModelBaseClass:
    def __init__(self, model_type, dsu_id, forecast_len=36, points_per_day=96, lag: int=None):
        """

        :param model_type:
        :param dsu_id:
        :param forecast_len:    foecast lenth in hours default 36 hours
        :param points_per_day:   number of points in a day default 96 -> 15min intervals
        :param lag:              lag in number of intervals default None
        """
        self.model_type = model_type
        self.model = None
        self.dsu_id = dsu_id
        self.foreast_len = forecast_len
        self.points_per_day = points_per_day
        self.lag = 0 if lag is None else lag
        if points_per_day < 24:
            raise ValueError('must have at least 24 points per day')
        self.points_per_hour = points_per_day / 24

    def build_model(self, **kwargs):
        pass

    def train(self, data_raw: pd.Series, **kwargs):
        pass

    def test(self, data_raw: pd.Series, **kwargs):
        pass

    def tune(self, data_raw: pd.Series, **kwargs):
        pass

    def predict(self, data_raw: pd.Series):
        pass

    def save(self, path=None):
        pass

    def load(self, path=None):
        pass


class KrrModelClass(ModelBaseClass):
    def __init__(self, dsu_id, forecast_len=36, points_per_day=96, lag: int=None, **kwargs):
        super().__init__('krr', dsu_id, forecast_len=forecast_len, points_per_day=points_per_day, lag=lag)
        self.alpha = kwargs.get('alpha')
        if self.alpha is None:
            self.alpha = 1
        self.gamma = kwargs.get('gamma')
        if self.gamma is None:
            self.gamma = 0.1
        # todo: implement an option to use a grid search to optimize hyperparameters for model
        self.grid_search = kwargs.get('grid_search')
        self.kernel = kwargs.get('kernel')
        if self.kernel is None:
            self.kernel = 'rbf'
        self.model = KernelRidge(
            alpha=self.alpha,
            gamma=self.gamma,
            kernel=self.kernel
        )

    def train(self, data_raw: pd.Series, **kwargs):
        x, y = _parse_data(data_raw, lag=self.lag, points_per_day=self.points_per_day)
        self.model.fit(x, y, sample_weight=kwargs.get('sample_weight'))

    def predict(self, data_raw: pd.Series):
        pred_vals = data_raw[-(self.lag+1):].copy()
        delta_time = datetime.timedelta(hours=1.0 / self.points_per_hour)
        for fcst_indx in range(0, int(self.foreast_len*self.points_per_hour)):
            ts = pred_vals.index[-1] + delta_time
            x = _parse_data_for_prediction(pred_vals, lag=self.lag, points_per_day=self.points_per_day)
            pred_val = self.model.predict(x)
            pred_vals[ts] = pred_val if len(pred_val) < 1 else pred_val[0]
        return pred_vals


def main():
    filename = 'UnitAvailabilities.csv'
    dsu_id = 'EE1'
    filepath = Path(__file__).parent
    data_raw = pd.read_csv(os.path.join(filepath, '..', 'data', filename), header=0)
    data_ts = create_timeseries_for_aggregate(data_raw, 'EE1')
    train_start = data_ts.index[0]
    train_start = datetime.datetime.strptime('2020-05-01 00:00:00', '%Y-%m-%d %H:%M:%S')
    train_end = datetime.datetime.strptime('2020-11-01 00:00:00', '%Y-%m-%d %H:%M:%S')
    model = KrrModelClass(dsu_id, points_per_day=48)
    model.train(data_ts[train_start: train_end])
    pred = model.predict(data_ts[train_end - datetime.timedelta(hours=4): train_end])
    print('done')
    pass


if __name__ == '__main__':
    main()
