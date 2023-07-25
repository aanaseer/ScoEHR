"""This file contains the code for loading the MIMIC-III and Hong datasets."""

import os
import pickle

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


class MIMIC3_ICD:
    def __init__(
        self, data_dir="data/mimic_data", data_file="mimic_processed_choi.matrix"
    ):
        """Loads the MIMIC-III dataset from the data directory."""
        data_path = os.path.join(data_dir, data_file)
        data = np.load(data_path, allow_pickle=True)
        self.dataset_full = torch.from_numpy(data)

    def data(self, use_train_test_split=True, test_size=0.30):
        if use_train_test_split:
            train_data, test_data = train_test_split(
                self.dataset_full, test_size=test_size, random_state=51
            )
            data_out = (train_data, test_data)
        else:
            data_out = self.dataset_full
        return data_out


class HongData:
    def __init__(
        self,
        data_dir="data/hong_data",
        data_file="data.npy",
        labels_dict="data_dict.pkl",
    ):
        """Loads the Hong dataset from the data directory."""
        data_path = os.path.join(data_dir, data_file)
        data_labels_path = os.path.join(data_dir, labels_dict)
        self.data_original = np.load(data_path).astype(np.float32)

        with open(data_labels_path, "rb") as f:
            data_dict = pickle.load(f)

        self.cts_indices, self.binary_indices = [], []

        for i in range(self.data_original.shape[1]):
            if data_dict[i]["data_type"] == "continuous":
                self.cts_indices.append(i)
            else:
                self.binary_indices.append(i)

        cts_data = self.data_original[:, self.cts_indices]
        binary_data = self.data_original[:, self.binary_indices]

        scaler = MinMaxScaler()
        self.fitted_scaler = scaler.fit(cts_data)
        transformed_data_cts = self.fitted_scaler.transform(cts_data)

        data_w_cts_scaling_and_binaries = np.hstack((transformed_data_cts, binary_data))

        self.data_labels_dict = np.load(data_labels_path, allow_pickle=True)

        self.labels = [
            self.data_labels_dict[i]["col_name"]
            for i in range(len(self.data_labels_dict))
        ]
        self.dataset_full = torch.from_numpy(data_w_cts_scaling_and_binaries)

    def data(self, use_train_test_split=True, test_size=0.30, device="cuda"):
        """Returns the data in the format (i.e. split or not) specified by the user."""
        if use_train_test_split:
            train_data, test_data = train_test_split(
                self.dataset_full, test_size=test_size, random_state=51
            )
            data_out = (train_data, test_data)
        else:
            data_out = self.dataset_full
        return data_out

    def reverse_minmaxscaling(self, scaled_data):
        """Reverses the minmax scaling applied to the data."""
        reverse = self.fitted_scaler.inverse_transform(scaled_data)
        return reverse

    def reverse_sin_cos_transformer(self, sin_component, cos_component, period):
        """Reverses the sin/cos transformation applied to the data."""
        x = np.mod(
            np.round(
                (np.degrees(np.arctan2(sin_component, cos_component))) / (360 / period)
            ),
            period,
        )
        return x

    def clean_up_data(self, data_set):
        """Converts the data into a readable format. (removes sin/cos components and reverses minmax scaling)"""
        cols_to_ignore = [
            "arrivalday_cos_component",
            "arrivalday_sin_component",
            "arrivalmonth_cos_component",
            "arrivalmonth_sin_component",
        ]

        for i in range(len(self.data_labels_dict)):
            if self.data_labels_dict[i]["data_type"] == "continuous":
                if (self.data_labels_dict[i]["col_name"] != "triage_vital_temp") and (
                    self.data_labels_dict[i]["col_name"] not in cols_to_ignore
                ):
                    data_set[:, i] = np.round(data_set[:, i])
                if self.data_labels_dict[i]["col_name"] == "triage_vital_temp" and (
                    self.data_labels_dict[i]["col_name"] not in cols_to_ignore
                ):
                    data_set[:, i] = np.round(data_set[:, i], 1)

        days_indices = [1, 2]
        month_indices = [3, 4]
        arrivalday_cos = data_set[:, days_indices[0]]
        arrivalday_sin = data_set[:, days_indices[1]]
        arrivalday_retrieved = self.reverse_sin_cos_transformer(
            arrivalday_cos, arrivalday_sin, period=7
        )

        arrivalmonth_cos = data_set[:, month_indices[0]]
        arrivalmonth_sin = data_set[:, month_indices[1]]
        arrivalmonth_retrieved = self.reverse_sin_cos_transformer(
            arrivalmonth_cos, arrivalmonth_sin, period=12
        )

        days_and_month_indices = days_indices + month_indices
        for i in days_and_month_indices:
            data_set = np.delete(data_set, 1, axis=1)

        data_set = np.insert(data_set, 1, arrivalday_retrieved, axis=1)
        data_set = np.insert(data_set, 2, arrivalmonth_retrieved, axis=1)

        updated_labels = list(
            filter(lambda label: label not in cols_to_ignore, self.labels)
        )
        updated_labels.insert(1, "arrival_day")
        updated_labels.insert(2, "arrival_month")

        return data_set, updated_labels
