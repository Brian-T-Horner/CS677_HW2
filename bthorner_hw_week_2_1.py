"""
Brian Horner
CS 677 - Summer 2
Date: 7/20/2021
Week 2 Homework Problems: ALL
This program works through stock data and uses prediction methods to return
results to the user.
"""

# Imports
import numpy as np
import pandas as pd
from function_helper import *
from table_printer import *
from matplotlib import pyplot as plt

"""Question 1.1 here for exporting reasons"""
# Reading csv files into pandas DataFrames
spy_df = pd.read_csv('SPY.csv')
hsy_df = pd.read_csv('HSY.csv')

# Adding column 'True Label and assigning '+' or '-' based on return
hsy_df['True Label'] = np.where(hsy_df['Return'] < 0, '-', '+')
spy_df['True Label'] = np.where(spy_df['Return'] < 0, '-', '+')

# Creating train and test dataframes
train_hsy_df = (hsy_df.loc[hsy_df['Year'].isin([2016, 2017, 2018])]).copy(
    deep=True)
train_spy_df = (spy_df.loc[spy_df['Year'].isin([2016, 2017, 2018])]).copy(
    deep=True)
test_hsy_df = (hsy_df.loc[hsy_df['Year'].isin([2019, 2020])]).copy(deep=True)
test_spy_df = (spy_df.loc[spy_df['Year'].isin([2019, 2020])]).copy(deep=True)


def following_ensemble(train_dataframe):
    """Calculates the total return from a $100 starting investment over
    2 years following the ensemble predictions."""
    investment = 100
    investment_list = [100]
    labels = (ensemble_learning(train_hsy_df, test_hsy_df))
    for index, value in enumerate(train_dataframe['Return']):
        try:
            for label in labels:
                if labels[index] == '+':
                    try:
                        investment = round(investment*(1+float(value)), 2)
                        investment_list.append(investment)
                    except ValueError:
                        pass
                    except IndexError:
                        pass
                elif label != '+':
                    pass
        except IndexError:
            pass
    return investment_list


def following_w(train_dataframe):
    """Calculates the total return from a $100 starting investment over
    2 years following the ensemble predictions."""
    investment = 100
    investment_list = [100]
    labels = (w_consecutive_prediction(train_hsy_df, test_hsy_df, 3))
    for index, value in enumerate(train_dataframe['Return']):
        try:
            for label in labels:
                if labels[index] == '+':
                    try:
                        investment = round(investment*(1+float(value)), 2)
                        investment_list.append(investment)
                    except ValueError:
                        pass
                    except IndexError:
                        pass
                elif label != '+':
                    pass
        except IndexError:
            pass
    return investment_list


def buy_and_hold(train_dataframe):
    """Calculates the total return from a $100 starting investment over
    2 years of the provide stock using a buy and hold strategy. Calculated
    from dataframe provided returns."""
    investment = 100
    investment_list = [100]
    train_dataframe = train_dataframe['Return']
    for value in train_dataframe:
        try:
            investment = round(investment*(1+float(value)), 2)
            investment_list.append(investment)
        except ValueError:
            pass
        except IndexError:
            pass
    return investment_list


if __name__ == '__main__':

    formatter = "---------------------------------------------------------------"

    """Question 1.1"""
    print("Training data for 'HSY' stock with 'True Label' column 'head view'.")
    print(train_hsy_df.head())
    print("\n")
    print("Testing data for 'HSY' stock with 'True Label' column 'head view'.")
    print(test_hsy_df.head())
    print(formatter)
    print("Training data for 'SPY' stock with 'True Label' column 'head view'.")
    print(train_spy_df.head())
    print("\n")
    print("Testing data for 'SPY' stock with 'True Label' column 'head view'.")
    print(test_spy_df.head())
    print(formatter)

    """Question 1.2"""
    print(f"--- Question 1.2 ---\n")
    print(f" The default probability of a 'up day' for the stock HSY is "
          f"{default_probability_pos(train_hsy_df)}%.")
    print(f" The default probability of a 'down day' for the stock HSY is "
          f"{default_probability_neg(train_hsy_df)}%.")
    print(f" The default probability of a 'up day' for the stock SPY is "
          f"{default_probability_pos(train_spy_df)}%.")
    print(f" The default probability of a 'down day' for the stock SPY is "
          f"{default_probability_neg(train_spy_df)}%.")
    print(formatter)

    """Question 1.3"""
    print(f"--- Question 1.3 ---\n")
    """HSY"""
    print(f"The probability of seeing an 'up day' after 3 days "
          f"of 'down days' for the stock 'HSY' is"
          f" {consecutive_days(train_hsy_df, 3, '-')}%")
    print(f"The probability of seeing an 'up day' after 2 days "
          f"of 'down days' for the stock 'HSY' is"
          f" {consecutive_days(train_hsy_df, 2, '-')}%")
    print(f"The probability of seeing an 'up day' after 1 day "
          f"of 'down days' for the stock 'HSY' is"
          f" {consecutive_days(train_hsy_df, 1, '-')}%")
    """SPY"""
    print(formatter)
    print(f"The probability of seeing an 'up day' after 3 days "
          f"of 'down days' for the stock 'SPY is"
          f" {consecutive_days(train_spy_df, 3, '-')}%")
    print(f"The probability of seeing an 'up day' after 2 days "
          f"of 'down days' for the stock 'SPY is"
          f" {consecutive_days(train_spy_df, 2, '-')}%")
    print(f"The probability of seeing an 'up day' after 1 day "
          f"of 'down days' for the stock 'SPY is"
          f" {consecutive_days(train_spy_df, 1, '-')}%")
    print(formatter)

    """Question 1.4"""
    print(f"--- Question 1.4 ---\n")
    """HSY"""
    print(f"The probability of seeing an 'up day' after 3 days "
          f"of 'up days' for the stock 'HSY is"
          f" {consecutive_days(train_hsy_df, 3, '+')}%")
    print(f"The probability of seeing an 'up day' after 2 days "
          f"of 'up days' for the stock 'HSY is"
          f" {consecutive_days(train_hsy_df, 3, '+')}%")
    print(f"The probability of seeing an 'up day' after 1 day "
          f"of 'up days' for the stock 'SPY is"
          f" {consecutive_days(train_hsy_df, 1, '+')}%")
    print(formatter)
    """SPY"""
    print(f"The probability of seeing an 'up day' after 3 days "
          f"of 'up days' for the stock 'SPY is"
          f" {consecutive_days(train_spy_df, 3, '+')}%")
    print(f"The probability of seeing an 'up day' after 2 days "
          f"of 'up days' for the stock 'SPY is"
          f" {consecutive_days(train_spy_df, 2, '+')}%")
    print(f"The probability of seeing an 'up day' after 1 days "
          f"of 'up days' for the stock 'SPY is"
          f" {consecutive_days(train_spy_df, 1, '+')}%")
    print(formatter)

    """Question 2.1"""
    print(f"--- Question 2.1 ---\n")
    test_hsy_df['W 2 Prediction'] = (w_consecutive_prediction(train_hsy_df,
                                                              test_hsy_df, 2))
    test_hsy_df['W 3 Prediction'] = (w_consecutive_prediction(train_hsy_df,
                                                              test_hsy_df, 3))
    test_hsy_df['W 4 Prediction'] = (w_consecutive_prediction(train_hsy_df,
                                                              test_hsy_df, 4))
    test_spy_df['W 2 Prediction'] = (w_consecutive_prediction(train_spy_df,
                                                              test_spy_df, 2))
    test_spy_df['W 3 Prediction'] = (w_consecutive_prediction(train_spy_df,
                                                              test_spy_df, 3))
    test_spy_df['W 4 Prediction'] = (w_consecutive_prediction(train_spy_df,
                                                              test_spy_df, 4))

    print("Testing data for 'HSY' stock with 2, 3, and 4 W pattern columns "
          "'head view'.")
    print(test_hsy_df.head())
    print("\n")
    print("Testing data for 'SPY' stock with 2, 3, and 4 W pattern columns "
          "'head view'.")
    print(test_spy_df.head())
    print(formatter)

    """Question 2.2"""
    print(f"---Question 2.2 ---\n")
    print("--W prediction accuracies for 'HSY'--")
    print(f"The accuracy of a two day (W) strategy on 'HSY' is "
          f"{w_calc_prediction_accuracy(train_hsy_df, test_hsy_df, 2)}%.")
    print(f"The accuracy of a three day (W) strategy on 'HSY' is "
          f"{w_calc_prediction_accuracy(train_hsy_df, test_hsy_df, 3)}%.")
    print(f"The accuracy of a four day (W) strategy on 'HSY' is "
          f"{w_calc_prediction_accuracy(train_hsy_df, test_hsy_df, 4)}%.")
    print("\n")
    print("--W prediction accuracies for 'SPY'--")
    print(f"The accuracy of a two day (W) strategy on 'SPY' is "
          f"{w_calc_prediction_accuracy(train_spy_df, test_spy_df, 2)}%.")
    print(f"The accuracy of a three day (W) strategy on 'SPY' is "
          f"{w_calc_prediction_accuracy(train_spy_df, test_spy_df, 3)}%.")
    print(f"The accuracy of a four day (W) strategy on 'SPY' is "
          f"{w_calc_prediction_accuracy(train_spy_df, test_spy_df, 4)}%.")
    print("\n")
    """Question 2.3"""
    print(f"---Question 2.3---\n")
    print(f"For the stock 'HSY' the three day (W) strategy gave the highest "
          f"amount of accurate predictions with "
          f"{w_calc_prediction_accuracy(train_hsy_df, test_hsy_df, 3)}%.")
    print(f"For the stock 'SPY' the four day (W) strategy gave the highest "
          f"amount of accurate predictions with "
          f"{w_calc_prediction_accuracy(train_spy_df, test_spy_df, 4)}%.")
    print("\n")
    """Question 3.1"""
    print(f"---Question 3.1---\n")
    test_hsy_df['Ensemble'] = ensemble_learning(train_hsy_df,
                                                       test_hsy_df)
    test_spy_df['Ensemble'] = ensemble_learning(train_spy_df, test_spy_df)

    print("Testing data for 'HSY' stock with Ensemble Learning labels 'head "
          "view'.")
    print(test_hsy_df.head())
    print("\n")
    print("Testing data for 'SPY' stock with Ensemble Learning labels 'head "
          "view'.")
    print(test_spy_df.head())
    print(formatter)

    """Question 3.2"""
    print(f"---Question 3.2---\n")
    print(f"Ensemble learning labels for the stock 'HSY' predicted "
          f"{ensemble_prediction_accuracy(train_hsy_df, test_hsy_df)}% of the "
          f"True Labels correctly.")
    print(f"Ensemble learning labels for the stock 'SPY' predicted "
          f"{ensemble_prediction_accuracy(train_spy_df, test_spy_df)}% of the "
          f"True Labels correctly.")
    print(formatter)


    """Question 3.3"""
    print(f"---Question 3.3---\n")
    print("--HSY Findings--")

    print(f"Ensemble learning labels for the stock 'HSY' predicted "
          f"{ensemble_prediction_accuracy(train_hsy_df, test_hsy_df, '-')}% of "
          f"the Negative True Labels correctly.")
    print(f"The a two day (W) strategy for stock 'HSY' predicted "
          f"{w_calc_prediction_accuracy(train_hsy_df, test_hsy_df, 2, '-')}%"
          f" of the Negative True Labels correctly")
    print(f"The a three day (W) strategy for stock 'HSY' predicted "
          f"{w_calc_prediction_accuracy(train_hsy_df, test_hsy_df, 3, '-')}%"
          f" of the Negative True Labels correctly")
    print(f"The a four day (W) strategy for stock 'HSY' predicted "
          f"{w_calc_prediction_accuracy(train_hsy_df, test_hsy_df, 4, '-')}%"
          f" of the Negative True Labels correctly")
    print("The Ensemble learning for 'HSY predicted Negative True Labels "
          "better than every W strategy but the four W day strategy.")
    print("\n")
    print("--SPY Findings--")
    print(f"Ensemble learning labels for the stock 'SPY' predicted "
          f"{ensemble_prediction_accuracy(train_spy_df, test_spy_df, '-')}% of "
          f"the Negative True Labels correctly.")
    print(f"The a two day (W) strategy for stock 'SPY' predicted "
          f"{w_calc_prediction_accuracy(train_spy_df, test_spy_df, 2, '-')}%"
          f" of the Negative True Labels correctly")
    print(f"The a three day (W) strategy for stock 'SPY' predicted "
          f"{w_calc_prediction_accuracy(train_spy_df, test_spy_df, 3, '-')}%"
          f" of the Negative True Labels correctly")
    print(f"The a four day (W) strategy for stock 'SPY' predicted "
          f"{w_calc_prediction_accuracy(train_spy_df, test_spy_df, 4, '-')}%"
          f" of the Negative True Labels correctly")
    print("The Ensemble learning for 'SPY' surprisingly and maybe incorrectly) "
          "was "
          "worse at predicting the Negative True Labels than all W strategies but"
          " the Three day in which it was tied.")
    print(formatter)


    """Question 3.4"""
    print(f"---Question 3.4---\n")
    """HSY"""
    print('--HSY Findings--')
    print(f"Ensemble learning labels for the stock 'HSY' predicted "
          f"{ensemble_prediction_accuracy(train_hsy_df, test_hsy_df, '+')}% of "
          f"the Positive True Labels correctly.")
    print(f"The a two day (W) strategy for stock 'HSY' predicted "
          f"{w_calc_prediction_accuracy(train_hsy_df, test_hsy_df, 2, '+')}%"
          f" of the Positive True Labels correctly")
    print(f"The a three day (W) strategy for stock 'HSY' predicted "
          f"{w_calc_prediction_accuracy(train_hsy_df, test_hsy_df, 3, '+')}%"
          f" of the Positive True Labels correctly")
    print(f"The a four day (W) strategy for stock 'HSY' predicted "
          f"{w_calc_prediction_accuracy(train_hsy_df, test_hsy_df, 4, '+')}%"
          f" of the Positive True Labels correctly")
    print("The Ensemble learning for 'HSY predicted Positive True Labels "
          "better than every W strategy but the three W day strategy.")

    """SPY"""
    print("\n")
    print("--SPY Findings--")
    print(f"Ensemble learning labels for the stock 'SPY' predicted "
          f"{ensemble_prediction_accuracy(train_spy_df, test_spy_df, '+')}% of "
          f"the Positive True Labels correctly.")
    print(f"The a two day (W) strategy for stock 'SPY' predicted "
          f"{w_calc_prediction_accuracy(train_spy_df, test_spy_df, 2, '+')}%"
          f" of the Positive True Labels correctly")
    print(f"The a three day (W) strategy for stock 'SPY' predicted "
          f"{w_calc_prediction_accuracy(train_spy_df, test_spy_df, 3, '+')}%"
          f" of the Positive True Labels correctly")
    print(f"The a four day (W) strategy for stock 'SPY' predicted "
          f"{w_calc_prediction_accuracy(train_spy_df, test_spy_df, 4, '+')}%"
          f" of the Positive True Labels correctly")
    print("The Ensemble learning for 'SPY predicted Positive True Labels "
           "better than every W strategy but the three W day strategy.")
    print(formatter)


    """Question 4.1 - 4.6 computations shown on table for Question 4.7"""

    """Question 4.7"""
    print(f"---Question 4.7---\n")
    table_print()
    print(formatter)

    print(f"---Question 4.8---\n")
    print("In general all prediction methods work better for predicting "
          "Positives in 'SPY' than 'HSY'. Although all prediction methods "
          "work better for predicting Negatives in 'HSY' than 'SPY'. ")



    """Question 5.1"""
    plt.plot(following_ensemble(train_hsy_df), 'r')
    plt.plot(buy_and_hold(train_hsy_df), 'b')
    plt.plot(following_w(train_hsy_df), 'g')
    plt.show()

    print(formatter)

    """Question 5.2"""
    print("--Question 5.2--")

    print(" I did not have enough time to mess around with the plot and make "
          "it actually readable as my values were astronomical.  I would assume "
          "that the W3 graph line has a higher growth as it had the best "
          "accuracy.")
