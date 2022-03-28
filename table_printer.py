"""
Brian Horner
CS 677 - Summer 2
Date: 7/20/2021
Week 2 Homework Problems:
This program formats the statistics for the machine learning done this week
in homeworks and formats it for printing in the required table.
"""
import numpy as np
import pandas as pd
from bthorner_hw_week_2_1 import train_hsy_df, test_hsy_df
from bthorner_hw_week_2_1 import train_spy_df, test_spy_df
from function_helper import statistics_calculator

def table_print():
    """Formats stock machine learning statistics into the required table."""
    master_stats_list = []
    header_list = ['W', 'Ticker', 'TP', 'FP', 'TN', 'FN', 'Accuracy',
                   'TPR', 'TNR']
    master_stats_list.append(statistics_calculator(train_hsy_df, test_hsy_df,
                                                   'W2', 'HSY'))
    master_stats_list.append(statistics_calculator(train_hsy_df, test_hsy_df,
                                                   'W3', 'HSY'))
    master_stats_list.append(statistics_calculator(train_hsy_df, test_hsy_df,
                                                   'W4', 'HSY'))
    master_stats_list.append(statistics_calculator(train_hsy_df, test_hsy_df,
                                                   'Ensemble', 'HSY'))

    master_stats_list.append(statistics_calculator(train_spy_df, test_spy_df,
                                                   'W2', 'SPY'))
    master_stats_list.append(statistics_calculator(train_spy_df, test_spy_df,
                                                   'W3', 'SPY'))
    master_stats_list.append(statistics_calculator(train_spy_df, test_spy_df,
                                                   'W4', 'SPY'))
    master_stats_list.append(statistics_calculator(train_spy_df, test_spy_df,
                                                   'Ensemble', 'SPY'))

    master_stats_list.insert(0, list(header_list))

    print(f"---Table of Stock Prediction with Machine Learning---")
    # Enumerating over print list
    for index, stuff in enumerate(master_stats_list):
        # Adding a | in front of each value of the lists in print list
        row = '|'.join(str(value).ljust(12) for value in stuff)
        # Printing the row for the list in print list
        print(row)
        # Adding a line between the header and the data rows
        if index == 0 or index == 4 or index ==8:
            print('-' * len(row))
    print("\n")




