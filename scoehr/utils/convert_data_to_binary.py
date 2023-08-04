"""Utility function to post-process data to binary."""


def convert_to_binary(data):
    data[data >= 0.5] = 1.0
    data[data < 0.5] = 0.0
    return data
