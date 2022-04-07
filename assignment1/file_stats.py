# Samson Nguyen
# 1001496565
# CSE 4392 Assignment 1
# Task 9

import numpy as np


def column_means(pathname):
    file = open(pathname)
    file_array = []
    averages = []
    for line in file:
        file_array.append(list(float(x) for x in line.split(',')))
    file.close()
    for col in range(len(file_array[0])):
        avg = 0
        for row in range(len(file_array)):
            avg += file_array[row][col]
        avg /= len(file_array)
        averages.append(avg)
    return np.asarray(averages)
