
import scipy.io
import numpy as np
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings('ignore')

class Service:
    def __init__(self):
        self.mat_file = []
        self.discharge_index_list = []
        self.discharge_df = pd.DataFrame()
        self.soh_list = []
        self.X_train = []
        self.X_test = []
        self.y_train = []
        self.y_test = []
        self.y_valid = []
        self.X_valid = []


    def load_mat_file(self, payload):
        print("mat 파일 불러오기 시작")
        filename = payload.context + payload.fname
        filename2 = payload.fname2
        mat_file = scipy.io.loadmat(filename)
        print("mat 파일 불러오기 완료")
        self.mat_file = mat_file[filename2]


    def cal_discharge_index_len(self):
        mat_file = self.mat_file
        self.discharge_index_list = [i for i in range(len(mat_file[0][0][0][0])) if
                                     mat_file[0][0][0][0][i][0][0] == 'discharge']

    def creat_values(self):
        mat_file = self.mat_file
        discharge_index_list = self.discharge_index_list
        discharge_values = np.array([(mat_file[0][0][0][0][i][3][0][0][0][0]).round(5) for i in discharge_index_list])
        cap_values = np.array([mat_file[0][0][0][0][i][3][0][0][6][0][0] for i in discharge_index_list])
        soh_values = np.array([(cap_values[i] / cap_values[0]).round(5) for i in range(len(cap_values))])
        time_values = np.array([(mat_file[0][0][0][0][i][3][0][0][5][0]).round(5) for i in discharge_index_list])

        discharge_df = pd.DataFrame()
        cap_df = pd.DataFrame()
        time_df = pd.DataFrame()
        soh_df = pd.DataFrame()

        for i in range(len(discharge_values)):
            discharge_df = pd.concat([discharge_df, pd.DataFrame(discharge_values[i])])
            time_df = pd.concat([time_df, pd.DataFrame(time_values[i])])
            for x in range(len(discharge_values[i])):
                cap_df = cap_df.append(pd.DataFrame(np.array(cap_values[i]).reshape(-1, )))
                soh_df = soh_df.append(pd.DataFrame(np.array(soh_values[i]).reshape(-1, )))

        merge_df = pd.DataFrame(columns=['vol','time', 'soh'])

        merge_df['vol'] = discharge_df[0].reset_index(drop=True)
        merge_df['time'] = time_df.reset_index(drop=True)
        merge_df['soh'] = soh_df.reset_index(drop=True)
        self.discharge_df = merge_df
        self.soh_list = list(set(list(soh_values)))

    def discharge_eda(self, payload):
        filename = payload.context + payload.fname
        discharge_df = self.discharge_df
        sns.lineplot(x=discharge_df.time, y=discharge_df.vol,  hue=discharge_df.soh)
        plt.savefig(filename)

    def make_data_set(self):
        discharge_df = self.discharge_df
        soh_list = self.soh_list
        temp_arr = np.array([])
        for soh_num in soh_list:
            temp_df = discharge_df[discharge_df.soh == soh_num]['vol']
            temp_arr = np.append(temp_arr,np.array(temp_df)[:100])
        self.make_data = temp_arr.reshape(len(soh_list),100,1)


    def creat_train_test_split(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.make_data,
                                                                                self.soh_list,
                                                                                test_size=0.2)
        self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(self.make_data,
                                                                                self.soh_list,
                                                                                test_size=0.2)

    def scaler_fit(self):
        scaler = MinMaxScaler()
        scr_X_train = scaler.fit_transform(self.X_train.reshape(-1,1)).reshape(self.X_train.shape[0],100,1)
        scr_X_valid = scaler.transform(self.X_valid.reshape(-1,1)).reshape(self.X_valid.shape[0],100,1)
        scr_X_test = scaler.transform(self.X_test.reshape(-1,1)).reshape(self.X_test.shape[0],100,1)
        return scr_X_train, scr_X_valid, scr_X_test


    def creat_lstm_model(self):
        scr_X_train, scr_X_valid, scr_X_test =self.scaler_fit()
        model = keras.models.Sequential([
            layers.LSTM(100, return_sequences=True, input_shape=[None, 1]),
            layers.LSTM(20, return_sequences=True),
            layers.Dense(1)
        ])
        model.compile(loss="mse",
                      optimizer="adam",
                      metrics=['mae'])
        #early_stopping_cb = keras.callbacks.EarlyStopping(patience=20)
        model_checkpoint_cb = keras.callbacks.ModelCheckpoint("./data/test.h5", save_best_only=True)
        #print(scr_X_train)
        print(scr_X_train.shape)


        history = model.fit(scr_X_train, np.array(self.y_train), epochs=100,
                            validation_data=(scr_X_valid, np.array(self.y_valid)),
                            batch_size =5,
                            callbacks=[model_checkpoint_cb]
                            )
        model.evaluate(scr_X_test, np.array(self.y_test))

