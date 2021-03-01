from src.common_utilities import *

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime
#import logging

POINTS_PER_DAY = 96
#LOGGER = logging.getLevelName()


class SimpleDataClass:
    def __init__(self, points_per_day=POINTS_PER_DAY):
        self.mean = np.zeros(points_per_day)
        self.var = np.zeros(points_per_day)
        self.stdev = np.zeros(points_per_day)
        self.min = np.zeros(points_per_day)

    def calc_stdev(self):
        self.stdev = np.sqrt(self.var)
        return self.stdev


class DataClass:
    def __init__(self, points_per_day=POINTS_PER_DAY):
        self.sum = np.zeros(points_per_day)
        self.sum2 = np.zeros(points_per_day)
        self.min = np.zeros(points_per_day) + 1E6
        self.points_per_day = points_per_day
        self.count = 0

    def add_day(self, data: np.array):
        if len(data) < self.points_per_day:
            print('data do not span a full day - not including in analysis')
            return
        for t_idx in range(0, self.points_per_day):
            self.min[t_idx] = min(self.min[t_idx], data[t_idx])
        self.sum += data
        self.sum2 += data * data
        self.count += 1
        return self.count

    def mean(self):
        if self.count > 0:
            return self.sum / self.count
        else:
            raise ValueError('not enough data for average to be calculated')

    def var(self):
        if self.count > 0:
            return (self.sum2 / self.count - self.mean()**2)
        else:
            raise ValueError('not enough data fore variance to be calculated')


class WeeklyDataClass:
    def __init__(self, points_per_day=POINTS_PER_DAY):
        self.monday = DataClass(points_per_day=points_per_day)
        self.tuesday = DataClass(points_per_day=points_per_day)
        self.wednesday = DataClass(points_per_day=points_per_day)
        self.thursday = DataClass(points_per_day=points_per_day)
        self.friday = DataClass(points_per_day=points_per_day)
        self.saturday = DataClass(points_per_day=points_per_day)
        self.sunday = DataClass(points_per_day=points_per_day)


class AggregateDataClass:
    def __init__(self, points_per_day=POINTS_PER_DAY):
        self.points_per_day = points_per_day
        self.monday = SimpleDataClass()
        self.tuesday = SimpleDataClass()
        self.wednesday = SimpleDataClass()
        self.thursday = SimpleDataClass()
        self.friday = SimpleDataClass()
        self.saturday = SimpleDataClass()
        self.sunday = SimpleDataClass()

    def add_site(self, site_data: WeeklyDataClass):
        days = Weekday
        for day in days:
            self.__dict__[day.name].mean += site_data.__dict__[day.name].mean()
            self.__dict__[day.name].var += site_data.__dict__[day.name].var()
            self.__dict__[day.name].min += site_data.__dict__[day.name].min

    def stdev(self):
        days = Weekday
        for day in days:
            self.__dict__[day.name].calc_stdev()


def load_profile(raw_data: pd.Series, points_per_day=POINTS_PER_DAY):
    data_by_weekday = {}
    for group in raw_data.groupby(raw_data.index.weekday):
        data_by_weekday.update({group[0]: group[1]})
    #data_by_weekday = [group[1] for group in raw_data.groupby(raw_data.index.weekday)]
    days = Weekday
    data = WeeklyDataClass(points_per_day)
    for weekday in days:
        data_by_day = [
            group[1] for group in data_by_weekday[weekday.value].groupby(data_by_weekday[weekday.value].index.date)
        ]
        for day in range(0, len(data_by_day)):
            data.__dict__[weekday.name].add_day(data_by_day[day].values)
    return data


def create_timeseries(data_raw: pd.DataFrame, site: str, market='POR'):
    data = pd.Series()
    flags = (data_raw[DataFrameKeys.site] == site) * (data_raw[DataFrameKeys.market] == market)
    indices = data_raw.index[flags]
    for index in indices:
        ts = datetime.datetime.strptime(data_raw.loc[index, DataFrameKeys.time], '%Y-%m-%d %H:%M:%S')
        data[ts] = data_raw.loc[index, DataFrameKeys.op_cert_const_avail]
    return data


def create_timeseries_for_aggregate(data_raw: pd.DataFrame, dsu_id: str):
    data = pd.Series()
    flags = data_raw[DataFrameKeys.dsu_id] == dsu_id
    indices = data_raw.index[flags]
    for index in indices:
        ts = datetime.datetime.strptime(data_raw.loc[index, DataFrameKeys.time], '%m/%d/%Y %H:%M')
        data[ts] = data_raw.loc[index, DataFrameKeys.esb_const_avail]
    return data


def build_aggregate(raw_data: pd.DataFrame, points_per_day=POINTS_PER_DAY, market='POR'):
    sites = raw_data[DataFrameKeys.site].unique()
    site_data_dict = {}
    aggregate = AggregateDataClass(points_per_day=points_per_day)
    for site in sites:
        ts_data = create_timeseries(raw_data, site, market=market)
        site_data_dict.update({site: load_profile(ts_data, points_per_day=points_per_day)})
        aggregate.add_site(site_data_dict[site])
    aggregate.stdev()
    return aggregate


def energy_bids(aggregate: AggregateDataClass, day: str):
    day = day.lower()
    ave = aggregate.__dict__[day].mean
    p95 = np.zeros(aggregate.points_per_day)
    p05 = np.zeros(aggregate.points_per_day)
    min_ = aggregate.__dict__[day].min
    hour = np.zeros(aggregate.points_per_day)
    for indx in range(0, len(ave)):
        p95[indx] = max(0, ave[indx] - 2.0 * aggregate.__dict__[day].stdev[indx])
        p05[indx] = ave[indx] + 2.0 * aggregate.__dict__[day].stdev[indx]
        hour[indx] = indx * 24 / aggregate.points_per_day
    plt.plot(hour, ave, label='ave')
    plt.plot(hour, p95, label='p95')
    plt.plot(hour, p05, label='p05')
    plt.plot(hour, min_, label='min')
    plt.ylabel('available power (kW)')
    plt.xlabel('hour of day')
    plt.title(day)
    plt.legend()
    plt.show()


def forecast_validation(data_ts: pd.Series, forecast: pd.Series, points_per_hour=4):
    freq = '{}S'.format(int(SECONDS_IN_HOUR / points_per_hour))
    points_per_day = HOURS_IN_DAY * points_per_hour
    basic_load_profile = load_profile(data_ts, points_per_day=points_per_day)
    today = data_ts.index[-1].replace(hour=0, minute=0, second=0, microsecond=0)
    start = today - datetime.timedelta(days=1)
    finish = today + datetime.timedelta(days=1)
    basic_load_time_index = pd.date_range(start=start, periods=3*points_per_day, freq=freq)
    days_of_week = Weekday
    yesterday_load = basic_load_profile.__dict__[days_of_week(start.weekday()).name]
    today_load = basic_load_profile.__dict__[days_of_week(today.weekday()).name]
    tomorrow_load = basic_load_profile.__dict__[days_of_week(finish.weekday()).name]
    ave_load = yesterday_load.mean().tolist() + today_load.mean().tolist() + tomorrow_load.mean().tolist()
    min_load = yesterday_load.min().tolist() + today_load.min().tolist() + tomorrow_load.min().tolist()
    ave_load_ts = pd.Series(ave_load, index=basic_load_time_index)
    min_load_ts = pd.Series(min_load, index=basic_load_time_index)
    ave_load_ts.plot(label='mean')
    min_load_ts.plot(label='min')
    forecast.plot(label='forecast')
    data_ts[start:].plot(label='actual')
    plt.legend()
    plt.ylabel('Expected Availability (kW)')


def main(day='monday', market='POR'):
    raw_data = pd.read_csv('~/Downloads/EE1_AvailabilityOct2020.csv', header=0)
    agg = build_aggregate(raw_data, market=market)
    energy_bids(agg, day)
    print('done')


if __name__ == '__main__':
    main()
