import numpy
import pandas
import iso8601
import datetime

from skgarden import RandomForestQuantileRegressor

# from sklearn.preprocessing import StandardScaler


FILENAME = ''


def read_data(filename: str, list_of_ids=None):
    """
    given a file name and list of id's, reads data from filename, generates and returns a dictionary of pandas
    DataFrames containing the data.
    :param filename:     filename with path
    :param list_of_ids:  list of ids to be read from file, if None defaults to [69, 75, 97]
    :return:             dictionary of DataFrames with id's as keys
    """
    header_list = ['id', 'power', 'consumption', 'capacity', 'expected_capacity', 'availibility', 'response_from_gen',
                   'response_from_dem', 'time_aggregated', 'sample_size']
    if list_of_ids is None:
        list_of_ids = [69, 75, 97]

    raw_data = pandas.read_csv(filename, header=None)
    for key in raw_data.keys():
        raw_data.rename(columns={key: header_list[key]}, inplace=True)
    data_dict = {}
    # initializing data dictionary
    key_dict = {}
    for key in raw_data.keys():
        if not (key in ['time_aggregated', 'id', 'sample_size']):
            key_dict.update({key: pandas.Series()})
    for id in list_of_ids:
        data_dict.update({id: pandas.DataFrame(key_dict)})

    # populating data dictionary
    for indx in raw_data.index:
        timestamp = iso8601.parse_date(raw_data.loc[indx, 'time_aggregated'])
        for key in key_dict:
            data_dict[raw_data.loc[indx, 'id']].loc[timestamp, key] = raw_data.loc[indx, key]

    return data_dict


def parse_data_for_training(df: pandas.DataFrame, dependent_var_str: str, length_of_lag=0, length_of_test=48):
    """
    separates data into arrays for dependent and independent variables
    :param length_of_test:
    :param df:                  pandas data frame containing data to be parsed
    :param dependent_var_str:   key (label) of dependent variable
    :param length_of_lag:       number of time intervals to be used as lag feature ('real-time' correction)
    :return:                    m x n-1 array of independent and m x 1 array of dependent variables where
                                df has m x n entries
    """
    independent_vars = []
    dependent_var = []
    for indx in range(length_of_lag, len(df) - length_of_test):
        # time dependence
        independent_vars_one_row = [df.index[indx].isocalendar()[1], df.index[indx].weekday(),
                                    df.index[indx].hour * 4 + df.index[indx].minute]
        # dependence on other variables
        #        for key in df.keys():
        #            if not(key == dependent_var_str):
        #                independent_vars_one_row.append(df.loc[df.index[indx], key])
        #        # setting lag
        #        for lag_indx in range(indx-length_of_lag, indx):
        #            independent_vars_one_row.append(df.loc[df.index[lag_indx], dependent_var_str])
        dependent_var.append(df.loc[df.index[indx], dependent_var_str])
        independent_vars.append(independent_vars_one_row)
    return numpy.array(independent_vars), numpy.array(dependent_var)


def parse_data_for_forecast(df: pandas.DataFrame, dependent_var_str, length_of_lag=48, length_of_forecast=48):
    """

    :param df:
    :param length_of_lag:
    :param length_of_forecast:
    :return:
    """
    independent_vars = []
    time_stamps = pandas.date_range(start=df.index[-1] - datetime.timedelta(minutes=15 * length_of_lag),
                                    periods=length_of_forecast + length_of_lag + 1,
                                    freq='15T')
    #    independent_vars_one_row = [df.index[-1].isocalendar()[1], df.index[-1].weekday(),
    #                                df.index[-1].hour*4 + df.index[-1].minute]
    #    # independent_vars = []
    #    for key in df.keys():
    #        if not(key == dependent_var_str):
    #            independent_vars_one_row.append(df.loc[df.index[-1], key])
    #
    #    for indx in range(0, length_of_lag):
    #        independent_vars_one_row.append(df.loc[df.index[indx-length_of_lag], dependent_var_str])

    #    return numpy.array([independent_vars_one_row])
    for ts in time_stamps:
        independent_vars.append([ts.isocalendar()[1], ts.weekday(), ts.hour * 4 + ts.minute])
    return numpy.array(independent_vars)


class ComponentForecast:
    def __init__(self, dependent_var_str: str, len_of_lag=48, len_of_forecast=48, min_samples_split=2,
                 len_of_test=48, n_estimators=1000, n_jobs=4):
        """
        initializing class
        :param dependent_var_str:  sets variable to be fit
        :param min_samples_split:  minimum number of samples needed to generate a new branch
        :param n_estimators:       number of estimators used
        """
        self.model = RandomForestQuantileRegressor(min_samples_split=min_samples_split, n_estimators=n_estimators,
                                                   bootstrap=True,
                                                   # min_weight_fraction_leaf=0.01, max_leaf_nodes=1000,
                                                   n_jobs=n_jobs)
        self.dependent_var = dependent_var_str
        self.length_of_lag = len_of_lag
        self.length_of_test = len_of_test
        self.length_of_forecast = len_of_forecast

    def train(self, df: pandas.DataFrame):
        x, y = parse_data_for_training(df, self.dependent_var, length_of_lag=self.length_of_lag,
                                       length_of_test=self.length_of_test)
        self.model.set_params(max_features=x.shape[1])
        self.model.fit(x, y)

    def test(self, df: pandas.DataFrame):
        #x = parse_data_for_forecast(df[:df.index[-self.length_of_test]], self.dependent_var,
        #                            length_of_lag=self.length_of_lag, length_of_forecast=self.length_of_forecast)
        #values = self.model.predict(x)
        #fcst = pandas.Series(values, df.index[-self.length_of_test:])
        fcst = self.predict(df[:df.index[-self.length_of_test]])
        diff = (fcst - df.loc[df.index[-self.length_of_test]:, self.dependent_var]) / \
               df.loc[df.index[-self.length_of_test]:, self.dependent_var]
        rms_err = numpy.sqrt(numpy.nanmean(diff ** 2))
        print(' RMS error: {}'.format(rms_err))
        return rms_err

    def predict(self, df: pandas.DataFrame, quantile=None):
        x = parse_data_for_forecast(df, self.dependent_var, length_of_lag=self.length_of_lag,
                                    length_of_forecast=self.length_of_forecast)
        values = self.model.predict(x, quantile=quantile)
        index = pandas.date_range(start=df.index[-1] - datetime.timedelta(minutes=15 * self.length_of_lag),
                                  periods=self.length_of_forecast + self.length_of_lag, freq='15T')
        fcast = pandas.Series(values[1:], index=index)
        if numpy.nansum(fcast) == 0:
            scale = 1
        else:
            scale = numpy.nansum(df.loc[index[0]:, self.dependent_var]) / numpy.nansum(fcast[:df.index[-1]])
        return scale * fcast[df.index[-1] + datetime.timedelta(minutes=15):]


class CompleteForecast:
    def __init__(self, data: pandas.DataFrame, lag_len=48, fcst_len=48, test_len=48,
                 min_samples_split=2, n_estimators=500, n_jobs=4):
        self.data = data
        self.models = {}
        for key in data.keys():
            if not (key in ['time_aggregated', 'id']):
                self.models.update({key: ComponentForecast(key, len_of_lag=lag_len, len_of_forecast=fcst_len,
                                                           len_of_test=test_len, min_samples_split=min_samples_split,
                                                           n_estimators=n_estimators, n_jobs=4)})

    def train(self, dep_var):
        self.models[dep_var].train(self.data)

    def train_all(self):
        for key in self.models.keys():
            self.train(key)

    def predict(self, dep_var, df, quantile=None):
        return self.models[dep_var].predict(df, quantile=quantile)

    def predict_all(self, df, quantile=None, length_of_fcast=48):
        data_dict = {}
        df_copy = df
        df_fcst = pandas.DataFrame()
        for key in self.models.keys():
            data_dict.update({key: self.predict(key, df_copy, quantile=quantile)})
        return pandas.DataFrame(data_dict)

    def test(self, dep_var):
        print(dep_var)
        return self.models[dep_var].test(self.data)

    def test_all(self):
        rms_err = {}
        for key in self.models.keys():
            rms_err.update({key: self.test(key)})
        return rms_err


def main():
    data = read_data('ds-challange-circuitdata.csv')
    model_69 = CompleteForecast(data[69], n_estimators=10)
    model_69.train_all()
    rms_err = model_69.test_all()
    res = model_69.predict_all(data[69])
    print('done')


if __name__ == '__main__':
    main()
