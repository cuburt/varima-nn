import math
import numpy as np
import pandas as pd
import datetime
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller, acf, pacf
from tsextract.feature_extraction.extract import build_features, build_features_forecast


class TimeSeriesPostProcessor:

    @staticmethod
    def scale_lagged_df(df, label_count):
        try:
            if label_count != 0:
                scaler_features = StandardScaler().fit(df[df.columns.values[:-label_count]])
                scaler_label = StandardScaler().fit(
                    np.array(df[df.columns.values[-label_count:]]).reshape(-1, label_count))
            else:
                scaler_features = StandardScaler().fit(df[df.columns.values])
                scaler_label = scaler_features

            return scaler_features, scaler_label

        except Exception as e:
            print(str(e))

    @staticmethod
    def inverse_transform(pred_series, last_actual_observation):
        try:
            series_undifferenced = pred_series.copy()
            series_undifferenced.iat[0] = series_undifferenced.iat[0] + last_actual_observation
            series_undifferenced = series_undifferenced.cumsum()

            return series_undifferenced
        except Exception as e:
            print(str(e))

    def get_forecast(self, model, range_start, range_end):
        df_pred = None
        feature_arrays = []
        features_requests = model.features_requests
        train_df = model.train_df
        scaler_label = model.scaler_label
        labels = model.labels
        date_column = model.date_column
        freq = model.freq
        forecast_dfs = []

        for label in labels:
            df = pd.DataFrame({date_column: train_df.index, label: train_df[label]})
            features_request = [fr for fr in features_requests if fr.get('name') == label][0]
            features_request_copy = features_request.copy()
            features_request_copy.pop("name")
            target_series = df[label]
            build_forecast_df = build_features_forecast(target_series, features_request_copy, include_tzero=True)
            forecast_dfs.append(build_forecast_df)

        tail = list(set([df.shape[0] for df in forecast_dfs]))
        tail = tail[0]

        for build_forecast_df in forecast_dfs:
            sub_scaler_features, sub_scaler_label = self.scale_lagged_df(build_forecast_df, 0)
            scaled_features_forecast = sub_scaler_features.transform(build_forecast_df[-tail:])
            feature_arrays.append(scaled_features_forecast)

        merged_array = np.concatenate(tuple(feature_arrays), axis=1)
        pred = model.predict(merged_array[:, :-(len(labels))])
        pred = scaler_label.inverse_transform(np.array(pred).reshape(-1, 1))

        forecast_range = pd.date_range(start=train_df.index[-1] + datetime.timedelta(days=1),
                                       end=train_df.index[-1] + datetime.timedelta(days=tail),
                                       freq=freq)
        print(forecast_range)
        forecast_range = forecast_range.to_list()

        for i, l in enumerate(labels):

            df_pred = pd.DataFrame({date_column: forecast_range})
            df_pred['pred_' + l] = np.ravel(pred[:tail, [i]])
            try:
                df_pred.set_index(date_column, inplace=True)
            except:
                pass
            target_series = df_pred['pred_' + l]
            print(target_series)
            volatility = target_series.groupby(target_series.index.day).std()
            forecast_vol = target_series.index.map(lambda d: volatility.loc[d.day])
            df_pred['forecast_vol'] = forecast_vol
            target_series = target_series * forecast_vol

            target_series = self.inverse_transform(target_series, train_df[l][-1])
            df_pred['pred_' + l] = target_series

            temp_range_start = datetime.datetime.strptime(range_start, '%Y-%m-%d')
            temp_range_end = datetime.datetime.strptime(range_end, '%Y-%m-%d')

            if temp_range_start and temp_range_end:
                df_pred = df_pred.loc[(df_pred.index >= temp_range_start) & (df_pred.index <= temp_range_end)]

            # plt.rcParams["figure.figsize"] = (30, 7)
            # # plt.scatter(train_df.index, train_df[l])
            # # plt.plot(train_df.index, train_df[l], label=l + ' actual')
            # plt.scatter(df_pred.index, df_pred['pred_' + l], label=l + ' forecast')
            # plt.plot(df_pred.index, df_pred['pred_' + l], label=l + ' forecast')
            # plt.legend()
            # plt.show()

        return np.array(df_pred)


class TimeSeriesPreProcessor:

    @staticmethod
    def segregate_datetime(df, date_col):
        try:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df['year'] = df[date_col].dt.year
            df['month'] = df[date_col].dt.month
            df['day'] = df[date_col].dt.day
            return df

        except Exception as e:
            print(str(e))

    @staticmethod
    def aggregate_interval(df, target_feature, freq='D'):
        try:
            sum_df = pd.DataFrame(df[target_feature].resample(freq).sum())
            sum_df.dropna(inplace=True)

            ave_df = pd.DataFrame(df[target_feature].resample(freq).mean())
            ave_df.dropna(inplace=True)

            if [pd.date_range(start=str(np.datetime_as_string(df.index.values[0], unit=freq)),
                              end=str(np.datetime_as_string(df.index.values[-1], unit=freq))).difference(
                df.index)] is not None:
                sum_df = sum_df.resample(freq).interpolate(method='linear')
                ave_df = ave_df.resample(freq).interpolate(method='linear')

            return sum_df, ave_df

        except Exception as e:
            print(str(e))

    @staticmethod
    def normalise_series(series):
        try:
            avg, dev = series.mean(), series.std()
            return (series - avg) / dev

        except Exception as e:
            print(str(e))

    @staticmethod
    def remove_increasing_vol(series, series_range='M'):
        transformed_series = series
        try:
            if series_range == 'D':
                volatility = series.groupby(series.index.day).std()
                vol = series.index.map(lambda d: volatility.loc[d.day])
                transformed_series = series / vol
            if series_range == 'M':
                volatility = series.groupby(series.index.month).std()
                vol = series.index.map(lambda d: volatility.loc[d.month])
                transformed_series = series / vol
            if series_range == 'Y':
                volatility = series.groupby(series.index.year).std()
                vol = series.index.map(lambda d: volatility.loc[d.year])
                transformed_series = series / vol
            return transformed_series

        except Exception as e:
            print(str(e))

    @staticmethod
    def difference(series):
        try:
            return series.diff()
        except Exception as e:
            print(str(e))

    @staticmethod
    def adfuller_test(values):
        # Augmented Dickey-Fuller test
        dickey_fuller_res = adfuller(values)
        print('test statistic: ', dickey_fuller_res[0], '\np-value: ', dickey_fuller_res[1], '\nstationary: ',
              str(dickey_fuller_res[1] <= 0.05))
        print('Critical Values:')
        for key, value in dickey_fuller_res[4].items():
            print('\t%s: %.3f' % (key, value))
        return dickey_fuller_res[1] <= 0.05

    @staticmethod
    def get_optimal_pq(series, alpha, lag_size):
        try:
            # plot_acf(series, alpha=alpha)
            acfs, aci = acf(series, nlags=math.ceil(len(series) * lag_size) - 1, alpha=alpha)
            acfs = [i for i in range(0, len(acfs)) if
                    acfs[i] < (np.array(aci[i]) - acfs[i])[0] or acfs[i] > (np.array(aci[i]) - acfs[i])[1]]
            # plot_pacf(series, alpha=alpha)
            pacfs, pci = pacf(series, nlags=math.ceil(len(series) * lag_size) - 1, alpha=alpha)
            pacfs = [i for i in range(0, len(pacfs)) if
                     pacfs[i] < (np.array(pci[i]) - pacfs[i])[0] or pacfs[i] > (np.array(pci[i]) - pacfs[i])[1]]
            return acfs[-1], pacfs[-1]
        except Exception as e:
            print(str(e))

    def create_features_request(self, target_col_df, label, window_size=False, diff_window_size=False, alpha=0.05,
                                lag_size=0.25):
        try:
            features_request = {}
            acfs, pacfs = self.get_optimal_pq(target_col_df, alpha, lag_size)
            window_size = window_size if window_size else pacfs
            diff_window_size = diff_window_size if diff_window_size else acfs
            if diff_window_size != 0:
                features_request["difference"] = [diff_window_size, 1]
            if window_size != 0:
                features_request["window"] = [window_size]
            else:
                features_request["window"] = [0]
            features_request['name'] = label

            return features_request

        except Exception as e:
            print(str(e))

    @staticmethod
    def create_lagged_df(target_col_df, features_request, target_lag=3, include_tzero=True):
        try:
            build_df = build_features(target_col_df, features_request, target_lag=target_lag,
                                      include_tzero=include_tzero)
            return build_df

        except Exception as e:
            print(str(e))

    @staticmethod
    def scale_lagged_df(df, label_count):
        try:
            if label_count != 0:
                scaler_features = StandardScaler().fit(df[df.columns.values[:-label_count]])
                scaler_label = StandardScaler().fit(
                    np.array(df[df.columns.values[-label_count:]]).reshape(-1, label_count))
            else:
                scaler_features = StandardScaler().fit(df[df.columns.values])
                scaler_label = scaler_features

            return scaler_features, scaler_label

        except Exception as e:
            print(str(e))

    def split_lagged_df(self, df, label_count, train_size=0.7):
        try:
            scaler_features, scaler_label = self.scale_lagged_df(df, label_count)
            scaled_features = scaler_features.transform(df[df.columns.values[:-label_count]])
            scaled_label = scaler_label.transform(
                np.array(df[df.columns.values[-label_count:]]).reshape(-1, label_count))

            # Split data using train proportion of 0.7
            train_size = int(scaled_features[:, :-1].shape[0] * train_size)

            X_train, y_train = scaled_features[:train_size, :-label_count], scaled_label[:train_size, :]
            X_test, y_test = scaled_features[train_size:, :-label_count], scaled_label[train_size:, :]

            return X_train, y_train, X_test, y_test

        except Exception as e:
            print(str(e))


    def preprocess_series(self, df, labels, date_column, freq, training_range):
        feature_dfs = []
        label_dfs = []
        features_requests = []
        df_data_copy = df.copy()
        for label in labels:
            print('processing label ' + label + '...')
            print(df_data_copy)
            df_data = pd.DataFrame({date_column: df_data_copy[date_column], label: df_data_copy[label]})
            print(df_data.head(10))
            initial_df = self.segregate_datetime(df_data, date_column)
            initial_df.set_index(date_column, inplace=True)
            print('created initial_df')
            print(initial_df.head(10))
            df_preprocessed_sum, df_preprocessed_ave = self.aggregate_interval(initial_df, label, freq)
            print('created df_preprocessed_ave')
            print(df_preprocessed_ave.head(10))
            df_preprocessed_ave = df_preprocessed_ave[:training_range]
            target_series = df_preprocessed_ave[label]
            target_series_for_transform = target_series.copy()
            target_series_for_transform = self.normalise_series(target_series_for_transform)
            print('normalised series')
            print(target_series_for_transform[10:])
            if not self.adfuller_test(pd.Series(target_series_for_transform.values)):
                target_series_for_transform = self.difference(target_series_for_transform)
                target_series_for_transform.fillna(0, inplace=True)
            print('tested stationarity of series')
            target_series_for_transform = self.remove_increasing_vol(target_series_for_transform, freq)
            print('transformed the series')
            print(target_series_for_transform[30:])
            features_request = self.create_features_request(target_series_for_transform, label)
            print('features_request:', features_request)
            print('created features request for series')
            features_request_for_transform = features_request.copy()
            features_request_for_transform.pop("name")
            lagged_df = self.create_lagged_df(target_series_for_transform, features_request_for_transform)
            print('created ARMA features for series')
            print(lagged_df.head(10))
            target_df = pd.DataFrame({label + '_target': lagged_df[lagged_df.columns.values[-1]].values},
                                     index=lagged_df.index)
            print('created dataframe for label')
            print(target_df.head(10))
            feature_df = lagged_df.drop(columns=lagged_df.columns.values[-1])
            print('dropped the label in series')
            feature_df.rename(columns={col: label + '_' + col for col in feature_df.columns.values if label not in col},
                              inplace=True)
            print('renamed ARMA features for series')
            print(feature_df.head(10))
            feature_df = pd.concat([feature_df, target_series], axis=1)
            features_requests.append(features_request)
            feature_dfs.append(feature_df)
            label_dfs.append(target_df)

        merged_feature_df = pd.concat(feature_dfs, axis=1)
        print('merged features from list')
        merged_label_df = pd.concat(label_dfs, axis=1)
        print('merged labels from list')
        train_df = pd.concat([merged_feature_df, merged_label_df], axis=1)
        print('created series for training')
        train_df.dropna(inplace=True, axis=0)
        print('dropped nas in the series')

        return train_df, features_requests