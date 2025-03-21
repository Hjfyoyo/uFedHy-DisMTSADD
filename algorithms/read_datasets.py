import pandas as pd
import torch.utils.data as data
import os
import numpy as np
import torch
from torch.utils.data import DataLoader,Dataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler

scaler = MinMaxScaler(feature_range=(0, 1))

class SMD_Dataset(data.Dataset):

    def __init__(self, dataidxs=None, train=True, transform=None, target_transform=None, download=False, window_len=5):

        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.window_len = window_len
        global scaler
        self.scaler = scaler

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):

        current_path = os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(os.getcwd()))))

        if self.train:
            train_path = current_path + '\\data\\datasets\\smd\\raw\\train'
            print(train_path)
            file_names = os.listdir(train_path)
            file_names.sort()
            data = []
            for i in range(len(file_names)):
                file_name = file_names[i]
                with open(train_path + '/' + file_name) as f:
                    this_data = pd.read_csv(train_path + '/' + file_name, header=None)
                    this_data = this_data.values.astype(np.float32)
                    data.append(this_data)
            data = np.concatenate(data, axis=0)
            data = self.scaler.fit_transform(data)
            target = data.copy()
        else:
            test_path = current_path + '\\data\\datasets\\smd\\raw\\test'
            file_names = os.listdir(test_path)
            file_names.sort()
            data = []
            for file_name in file_names:
                with open(test_path + '/' + file_name) as f:
                    this_data = []
                    for line in f.readlines():
                        this_data.append(line.split(','))
                    this_data = np.asarray(this_data)
                    this_data = this_data.astype(np.float32)
                data.append(this_data)
            data = np.concatenate(data, axis=0)
            data = self.scaler.transform(data)
            test_target_path = current_path + '\\data\\datasets\\smd\\raw\\test_label'
            file_names = os.listdir(test_target_path)
            file_names.sort()
            target = []
            for file_name in file_names:
                with open(test_target_path + '/' + file_name) as f:
                    this_target = []
                    for line in f.readlines():
                        this_target.append(line.split(','))
                    this_target = np.asarray(this_target)
                    this_target = this_target.astype(np.float32)
                target.append(this_target)
            target = np.concatenate(target, axis=0)

        if self.dataidxs:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if index + 1 - self.window_len < 0:
            data = self.data[0: index + 1]
            delta = self.window_len - data.shape[0]
            data0 = data[0]
            if len(data0.shape) == 1:
                data0 = data0[np.newaxis, :]
            data0 = np.repeat(data0, delta, axis=0)
            # print(data0.shape, data.shape)
            data = np.concatenate((data0, data), axis=0)
        else:
            data = self.data[index + 1 - self.window_len: index + 1]
        target = self.target[index]

        return data, target

    def __len__(self):
        return self.data.shape[0]

class SMAP_Dataset(data.Dataset):

    def __init__(self, dataidxs=None, train=True, transform=None, target_transform=None, download=False, window_len=5):

        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.window_len = window_len
        global scaler
        self.scaler = scaler

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):

        current_path = os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(os.getcwd()))))

        if self.train:
            train_path = current_path + '\\data\\datasets\\smap\\raw'
            data = np.load(train_path + '\\train.npy')
            data = data.astype(np.float32)
            target = data.copy()
        else:
            test_path = current_path + '\\data\\datasets\\smap\\raw'
            data = np.load(test_path + '\\test.npy')
            data = data.astype(np.float32)
            test_target_path = current_path + '\\data\\datasets\\smap\\raw'
            target = np.load(test_target_path + '\\test_label.npy')
            target = target.astype(np.float32)

        if self.dataidxs:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if index + 1 - self.window_len < 0:
            data = self.data[0: index + 1]
            delta = self.window_len - data.shape[0]
            data0 = data[0]
            if len(data0.shape) == 1:
                data0 = data0[np.newaxis, :]
            data0 = np.repeat(data0, delta, axis=0)
            # print(data0.shape, data.shape)
            data = np.concatenate((data0, data), axis=0)
        else:
            data = self.data[index + 1 - self.window_len: index + 1]
        target = self.target[index]

        return data, target

    def __len__(self):
        return self.data.shape[0]

class PSM_Dataset(data.Dataset):

    def __init__(self, dataidxs=None, train=True, transform=None, target_transform=None, download=False, window_len=5):

        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.window_len = window_len
        global scaler
        self.scaler = scaler

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):

        current_path = os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(os.getcwd()))))

        if self.train:
            train_path = current_path + '\\FedTADBench-main\\data\\datasets\\psm\\raw'
            data = pd.read_csv(train_path + '\\train.csv')
            data.drop(columns=[r'timestamp_(min)'], inplace=True)
            data = data.values.astype(np.float32)
            data = np.nan_to_num(data)
            data = self.scaler.fit_transform(data)
            target = data.copy()
        else:
            test_path = current_path + '\\FedTADBench-main\\\data\\datasets\\psm\\raw'
            file_names = os.listdir(test_path)
            file_names.sort()
            data = pd.read_csv(test_path + '\\test.csv')
            data.drop(columns=[r'timestamp_(min)'], inplace=True)
            data = data.values.astype(np.float32)
            data = np.nan_to_num(data)
            data = self.scaler.transform(data)
            test_target_path = current_path + '\\FedTADBench-main\\data\\datasets\\psm\\raw\\test_label.csv'
            target_csv = pd.read_csv(test_target_path)
            target_csv.drop(columns=[r'timestamp_(min)'], inplace=True)
            target = target_csv.values
            target = target.astype(np.float32)

        if self.dataidxs:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if index + 1 - self.window_len < 0:
            data = self.data[0: index + 1]
            delta = self.window_len - data.shape[0]
            data0 = data[0]
            if len(data0.shape) == 1:
                data0 = data0[np.newaxis, :]
            data0 = np.repeat(data0, delta, axis=0)
            # print(data0.shape, data.shape)
            data = np.concatenate((data0, data), axis=0)
        else:
            data = self.data[index + 1 - self.window_len: index + 1]

        # target = self.target[index]
        target = np.float32(self.target[index:index + self.window_len])
        # target  = target.shape(-1,self.window_len)

        return data, target

    def __len__(self):
        return self.data.shape[0]

class MSL_Dataset(data.Dataset):

    def __init__(self, dataidxs=None, train=True, transform=None, target_transform=None, download=False, window_len=5):

        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.window_len = window_len
        global scaler
        self.scaler = scaler

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):

        current_path = os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(os.getcwd()))))

        if self.train:
            train_path = current_path + '\\data\\datasets\\MSL\\raw'
            data = np.load(train_path + '\\train.npy')
            data = data.astype(np.float32)
            target = data.copy()
        else:
            test_path = current_path + '\\data\\datasets\\MSL\\raw'
            data = np.load(test_path + '\\test.npy')
            data = data.astype(np.float32)
            test_target_path = current_path + '\\data\\datasets\\MSL\\raw'
            target = np.load(test_target_path + '\\test_label.npy')
            target = target.astype(np.float32)

        if self.dataidxs:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if index + 1 - self.window_len < 0:
            data = self.data[0: index + 1]
            delta = self.window_len - data.shape[0]
            data0 = data[0]
            if len(data0.shape) == 1:
                data0 = data0[np.newaxis, :]
            data0 = np.repeat(data0, delta, axis=0)
            # print(data0.shape, data.shape)
            data = np.concatenate((data0, data), axis=0)
        else:
            data = self.data[index + 1 - self.window_len: index + 1]
        target = self.target[index]

        return data, target

    def __len__(self):
        return self.data.shape[0]

class MSLSegLoader(Dataset):
    def __init__(self, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(os.path.join(root_path, "test_label.npy"))
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step

        if self.flag == "train":

            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])

class SWaTSegLoader(Dataset):
    def __init__(self, root_path, win_size, step=1, flag="train"):
        current_path = 'E:\\pythonProject\\FedTAD-main'
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        train_path = current_path + '\\data\\datasets\\swat\\raw'
        train_data = pd.read_csv(train_path + '/swat_train2.csv')
        train_data = train_data.values[:, :-1]
        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        self.train = train_data

        test_path = current_path + '\\data\\datasets\\swat\\raw'
        tdata = pd.read_csv(test_path + '\\swat2.csv')
        test_data = tdata.values[:, :-1]
        self.scaler.fit(test_data)
        test_data = self.scaler.transform(test_data)
        self.test = test_data

        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = tdata.values[:, -1:]


        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step

        if self.flag == "train":

            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])

if __name__ == '__main__':
    # psm_train = PSM_Dataset()
    # psm_test = PSM_Dataset(train=False)
    data_set = MSLSegLoader(root_path = 'E:\\pythonProject\\FedTADBench-main\\data\\datasets\\msl\\raw',win_size = 5,flag='test')

