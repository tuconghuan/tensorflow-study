import numpy as np
import pandas as pd
import pdb
import json
import os


class DataLoader():
    """A class for loading and transforming data for the lstm model"""

    def __init__(self, filename, code, date, split, cols):

        self.filename = filename
        self.code  = code
        self.date  = date
        self.split = split
        self.cols  = cols

    def get_data(self):
        df_ori = pd.read_csv(self.filename)#, encoding='gb18030',index_col=0)

        if "0" in self.code:   # 取所有日期
            df_code = df_ori
        else:
            df_code = df_ori[df_ori['Code'].isin(self.code)]  # 取特定代码股票

        if "0" in self.date:  # 取所有日期
            df = df_code
        else:
            df = df_code[df_ori['Date'].isin(self.date)]

        return df

    def _normalise_window(self, data_window, normalise):
        if normalise == 0:
            data_window_n = data_window
            data_window['label'] = data_window['change']

        if normalise == 1:   # 相对首时刻的变化来归一化
            for col in self.cols:
                data_window[(col+'_change')] = data_window.loc[:, col] / data_window.loc[:, col][0] - 1

            data_window['label'] = data_window['price_change']

            data_window_n = data_window.drop(self.cols, axis=1)

        if normalise == 2:   # 相对前一时刻的变化来归一化
            for col in self.cols:
                # pdb.set_trace()
                data_window[(col+'_change')] = (data_window.loc[:, col] - data_window.loc[:, col].shift(1)) \
                                               / data_window.loc[:, col].shift(1)
                data_window = data_window.fillna(0)

            data_window['label'] = data_window['price_change']
            data_window_n = data_window.drop(self.cols, axis=1)

        if normalise == 3:   #

            data_window_n = pd.DataFrame(columns=["label"])
            data_window_n['label'] = data_window['change']
            data_window_n[data_window_n > 0] = 1
            data_window_n[data_window_n < 0] = -1

            d_tmp = data_window.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
            for col in self.cols:
                data_window_n[(col + '_change')] = d_tmp[col]

        return data_window_n

    def get_train_test_data(self, seq_len, normalise):

        df = self.get_data()
        data_train = df.get(self.cols).iloc[0: int(df.shape[0]*self.split), :]
        data_test = df.get(self.cols).iloc[int(df.shape[0]*self.split):, :]
        len_train = len(data_train)
        len_test = len(data_test)
        # pdb.set_trace()

        train_x = []
        train_y = []
        test_x = []
        test_y = []
        train_empty_num = 0
        train_null_num = 0
        train_len_num = 0

        for i in range(len_train - seq_len):
            if i%1000 == 0:
                print("prepare train data %d " % i)

            data_train_window = data_train.iloc[i:i+seq_len]
            data_train_window_n = self._normalise_window(data_train_window, normalise)

            # x = data_train_window_n.iloc[0:-1, :]
            # y = data_train_window_n.iloc[-1:, :1]
            x = data_train_window_n.iloc[0:-1, 1:]
            y = data_train_window_n.iloc[-1:, :1]

            if x.empty or y.empty:
                train_empty_num += 1
                continue
            if np.any(x.isnull()):
                train_null_num += 1
                continue
            if x.shape[0] < seq_len-1:
                train_len_num += 1
                continue

            train_x.append(x.values)
            train_y.append(y.values)

        print("train_empty_num : %d" % train_empty_num)
        print("train_null_num : %d" % train_null_num)
        print("train_len_num : %d" % train_len_num)

        print("train data done")

        for i in range(len_test - seq_len):
            if i%1000 == 0:
                print("prepare test data %d " % i)

            data_test_window = data_test.iloc[i:i+seq_len]
            data_test_window_n = self._normalise_window(data_test_window, normalise)

            # x = data_test_window_n.iloc[0:-1, :]
            # y = data_test_window_n.iloc[-1:, :1]
            x = data_test_window_n.iloc[0:-1, 1:]
            y = data_test_window_n.iloc[-1:, :1]
            if x.empty or y.empty:
                continue
            if np.any(x.isnull()):
                continue
            if x.shape[0] < seq_len-1:
                continue

            test_x.append(x.values)
            test_y.append(y.values)
        print("test data done")

        return np.array(train_x), np.array(train_y), np.array(test_x), np.array(test_y)


if __name__ == '__main__':

    configs = json.load(open('./config/config', 'r'))

    data = DataLoader(
        os.path.join('data', configs['data']['filename']),
        configs['data']['code'],
        configs['data']['date'],
        configs['data']['train_test_split'],
        configs['data']['columns']
    )

    train_x, train_y, test_x, test_y = data.get_train_test_data(seq_len=10, normalise=3)

    np.save("train_x", train_x)
    np.save("train_y", train_y)
    np.save("test_x", test_x)
    np.save("test_y", test_y)
