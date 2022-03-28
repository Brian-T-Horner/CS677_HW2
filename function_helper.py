"""
Brian Horner
CS 677 - Summer 2
Date: 7/20/2021
Week 2 Homework Problems:
This program
"""

def default_probability_pos(dataframe):
    """Calculates the default probability of having a positive day in the
    given stock data. Returns probability result as a percentage."""
    input_df = dataframe
    input_df_count = input_df['True Label'].count()
    df_positive_list = [label for label in input_df['True Label'] if
                        label == '+']
    return round((len(df_positive_list) / input_df_count) * 100, 2)


def default_probability_neg(dataframe):
    """Calculates the default probability of having a negative dy in the
    given stock data. Returns probability result as a percentage."""
    input_df = dataframe
    input_df_count = input_df['True Label'].count()
    df_negative_list = [label for label in input_df['True Label'] if
                        label == '-']

    return round((len(df_negative_list) / input_df_count) * 100, 2)


def consecutive_days(dataframe, days, pos_or_neg):
    """Calculates the probability that after seeing either 1, 2,
    or 3 consecutive days (up or down) that you will have a positive day.
    Returns probability result as a percentage. Additional functionality to
    calculate having a negative day next for use in other functions."""
    true_labels = dataframe['True Label'].values
    pos_seq_count = 0
    neg_seq_count = 0
    total_seq_count = 0
    try:
        for index, value in enumerate(true_labels):
            if days == 3:
                if value == pos_or_neg and true_labels[index+1] == value and \
                        true_labels[index + 2] == value:
                    total_seq_count += 1
                    if true_labels[index+3] == '+':
                        pos_seq_count += 1
                    elif true_labels[index+3] == '-':
                        neg_seq_count += 1
            if days == 2:
                if value == pos_or_neg and true_labels[index+1] == value:
                    total_seq_count += 1
                    if true_labels[index+2] == '+':
                        pos_seq_count += 1
                    elif true_labels[index+2] == '-':
                        neg_seq_count += 1
            if days == 1:
                if value == pos_or_neg:
                    total_seq_count += 1
                    if true_labels[index+1] == '+':
                        pos_seq_count += 1
                    elif true_labels[index+1] == '-':
                        neg_seq_count += 1
        else:
            pass
        # Will change data calculated at the end of the dataframe
    except IndexError:
        pass
    if pos_or_neg == '+':
        return round((pos_seq_count/total_seq_count)*100, 2)
    elif pos_or_neg == '-':
        return round((pos_seq_count/total_seq_count)*100, 2)


def w_consecutive_prediction(train_dataframe, test_dataframe, w_value):
    """Calculates a predicted return list for the testing dataframe based
    off how many times a pattern is observed in the training dataframe.
    Each element in the training_dataframe is exampled for with a pattern of
    2, 3, or 4, days and the pattern is then matched against the training
    dataframe."""
    true_labels_train = train_dataframe['True Label'].values
    true_labels_test = test_dataframe['True Label'].values
    testing_list = true_labels_train.tolist()
    testing_string = str(testing_list)
    testing_string = testing_string.replace("[", "").replace("]", "")
    neg_count_list = []
    pos_count_list = []
    resulting_values = []
    pos_pattern_list = []
    neg_pattern_list = []
    days_to_check = w_value
    for test_index, test_value in enumerate(true_labels_test):
        if days_to_check == 2:
            value1 = true_labels_test[test_index-1]
            value2 = true_labels_test[test_index-2]
            pos_pattern = [value2, value1, '+']
            neg_pattern = [value2, value1, '-']
            pos_pattern_list.append(pos_pattern)
            neg_pattern_list.append(neg_pattern)
        elif days_to_check == 3:
            value1 = true_labels_test[test_index-1]
            value2 = true_labels_test[test_index-2]
            value3 = true_labels_test[test_index-3]
            pos_pattern = [value3, value2, value1, '+']
            neg_pattern = [value3, value2, value1, '-']
            pos_pattern_list.append(pos_pattern)
            neg_pattern_list.append(neg_pattern)
        elif days_to_check == 4:
            value1 = true_labels_test[test_index-1]
            value2 = true_labels_test[test_index-2]
            value3 = true_labels_test[test_index-3]
            value4 = true_labels_test[test_index-4]
            pos_pattern = [value4, value3, value2, value1, '+']
            neg_pattern = [value4, value3, value2, value1, '-']
            pos_pattern_list.append(pos_pattern)
            neg_pattern_list.append(neg_pattern)
    for neg_pattern in neg_pattern_list:
        neg_pattern = str(neg_pattern)
        neg_pattern = neg_pattern.replace("[", "").replace("]", "")
        neg_pattern_count = testing_string.count(neg_pattern)
        neg_count_list.append(neg_pattern_count)
    for pos_pattern in pos_pattern_list:
        pos_pattern = str(pos_pattern)
        pos_pattern = pos_pattern.replace("[", "").replace("]", "")
        pos_pattern_count = testing_string.count(pos_pattern)
        pos_count_list.append(pos_pattern_count)
    for index, pos_value in enumerate(pos_count_list):
        if pos_value > neg_count_list[index]:
            resulting_values.append('+')
        elif pos_value < neg_count_list[index]:
            resulting_values.append('-')
        else:
            if default_probability_pos(train_dataframe) > \
                        default_probability_neg(train_dataframe):
                resulting_values.append('+')
            elif default_probability_neg(train_dataframe) > \
                    default_probability_pos(train_dataframe):
                resulting_values.append('-')
            else:
                print("Error this condition should not be reached.")

    return resulting_values


def w_calc_prediction_accuracy(train_dataframe, test_dataframe, w_value,
                               label='ALL', returns='Percent'):
    """Calculates the accuracy of the selected consecutive days (W) pattern
    Can return the percent of accuracy for all, positives, or negatives or
    return the count of accurate total labels, positive labels, or negative."""
    accuracy_dict = {'Positive': 0, 'Negative': 0}
    correct_count = 0
    incorrect_count = 0
    true_labels_test = test_dataframe['True Label'].values
    true_labels_test = true_labels_test.tolist()
    comparison_labels = w_consecutive_prediction(train_dataframe,
                                                 test_dataframe, w_value)

    for index, true_value in enumerate(true_labels_test):
        if true_value == comparison_labels[index]:
            correct_count += 1
            if comparison_labels[index] == '+':
                accuracy_dict['Positive'] += 1
            else:
                accuracy_dict['Negative'] += 1
        elif true_value != comparison_labels[index]:
            incorrect_count +=1
        else:
            print("Error this condition should not be reached.")
    if returns == 'Percent':
        if label == 'ALL':
            return round((correct_count / (len(comparison_labels)))*100, 2)
        elif label == '+':
            return round((accuracy_dict['Positive'] / (len(
                comparison_labels)))*100, 2)
        elif label == '-':
            return round((accuracy_dict['Negative'] / (len(
                comparison_labels)))*100, 2)
    elif returns == 'Value':
        if label == 'ALL':
            return accuracy_dict['Positive'] + accuracy_dict['Negative']
        elif label == '+':
            return accuracy_dict['Positive']
        elif label == '-':
            return accuracy_dict['Negative']


def false_pos_neg_w(train_dataframe, test_dataframe, w, label='ALL'):
    """Calculates the number of false positive and false negatives from a
    selected consecutive days (W) pattern machine predicting method. """
    incorrect_dict = {'Positive': 0, 'Negative': 0}
    true_labels_test = test_dataframe['True Label'].values
    true_labels_test = true_labels_test.tolist()
    comparison_labels = w_consecutive_prediction(train_dataframe,
                                                 test_dataframe, w)
    for index, true_value in enumerate(true_labels_test):
        if true_value == comparison_labels[index]:
            pass
        elif true_value != comparison_labels[index]:
            if comparison_labels[index] == '-':
                incorrect_dict['Negative'] += 1
            elif comparison_labels[index] == '+':
                incorrect_dict['Positive'] +=1
            else:
                print('Error this condition should not be met.')
        else:
            print("Error this condition should not be reached.")
    if label == 'ALL':
        return incorrect_dict['Positive'] + incorrect_dict['Negative']
    elif label == '+':
        return incorrect_dict['Positive']
    elif label == '-':
        return incorrect_dict['Negative']
    else:
        print('Error: This condition should not be met.')


def ensemble_learning(train_dataframe, test_dataframe):
    """Predicts the return of a testing dataset based off 3 pattern
    prediction methods. Determines the return based on the mode of the
    predicting methods."""
    two_day_list = w_consecutive_prediction(train_dataframe, test_dataframe, 2)
    three_day_list = w_consecutive_prediction(train_dataframe,
                                              test_dataframe, 3)
    four_day_list = w_consecutive_prediction(train_dataframe, test_dataframe, 4)
    ensemble_returns = []
    for index, two_value in enumerate(two_day_list):
        temp_list = [two_value, three_day_list[index], four_day_list[index]]
        pos_count = 0; neg_count = 0
        for element in temp_list:
            if element == '+':
                pos_count += 1
            else:
                neg_count += 1
        if pos_count > neg_count:
            ensemble_returns.append('+')
        else:
            ensemble_returns.append('-')
    return ensemble_returns


def ensemble_prediction_accuracy(train_dataframe, test_dataframe,
                                 label='ALL', returns='Percent'):
    """Calculates the accuracy of the ensemble prediction method. Can return
    the percent of accuracy for all, positives, or negatives or return the
    count of accurate total labels, positive labels, or negative."""
    accuracy_dict = {'Positive': 0, 'Negative': 0}
    correct_count = 0
    incorrect_count = 0
    true_labels_test = test_dataframe['True Label'].values
    true_labels_test = true_labels_test.tolist()
    comparison_labels = ensemble_learning(train_dataframe, test_dataframe)

    for index, true_value in enumerate(true_labels_test):
        if true_value == comparison_labels[index]:
            correct_count += 1
            if comparison_labels[index] == '+':
                accuracy_dict['Positive'] += 1
            else:
                accuracy_dict['Negative'] += 1
        elif true_value != comparison_labels[index]:
            incorrect_count += 1
        else:
            print("Error this condition should not be reached.")
    if returns == 'Percent':
        if label == 'ALL':
            return round((correct_count / (len(comparison_labels)))*100, 2)
        elif label == '+':
            return round((accuracy_dict['Positive'] / (len(
                comparison_labels)))*100, 2)
        elif label == '-':
            return round((accuracy_dict['Negative'] / (len(
                comparison_labels)))*100, 2)
        else:
            print("Error this condition should not be hit in ensemble prediction "
                  "accuracy.")
    elif returns == 'Value':
        if label == 'ALL':
            return accuracy_dict['Positive'] + accuracy_dict['Negative']
        elif label == '+':
            return accuracy_dict['Positive']
        elif label == '-':
            return accuracy_dict['Negative']
    else:
        print("Error this condition should not be hit in ensemble prediction "
                  "accuracy.")


def false_pos_neg_ensemble(train_dataframe, test_dataframe, label='ALL'):
    """This function determines the false negatives and false positives from
    an ensemble learning method from the provided test and training
    dataframes."""
    incorrect_dict = {'Positive': 0, 'Negative': 0}
    true_labels_test = test_dataframe['True Label'].values
    true_labels_test = true_labels_test.tolist()
    comparison_labels = ensemble_learning(train_dataframe, test_dataframe)
    for index, true_value in enumerate(true_labels_test):
        if true_value == comparison_labels[index]:
            pass
        elif true_value != comparison_labels[index]:
            if comparison_labels[index] == '-':
                incorrect_dict['Negative'] += 1
            elif comparison_labels[index] == '+':
                incorrect_dict['Positive'] +=1
            else:
                print('Error this condition should not be met.')
        else:
            print("Error this condition should not be reached.")
    if label == 'ALL':
        return incorrect_dict['Positive'] + incorrect_dict['Negative']
    elif label == '+':
        return incorrect_dict['Positive']
    elif label == '-':
        return incorrect_dict['Negative']
    else:
        print('Error: This condition should not be met.')


def statistics_calculator(train_dataframe, test_dataframe, method, stock):
    """This function collates statistics about the selected machining
    learning method. """
    statistics_dictionary = {'Method': method, 'Stock': stock, 'TP': 0, 'FP': 0,
                             'TN': 0, 'FN': 0, 'Accuracy': 0, 'TPR': 0,
                             'TNR': 0}
    if method == 'Ensemble':
        # TP - True Positives
        statistics_dictionary['TP'] = ensemble_prediction_accuracy(
            train_dataframe, test_dataframe, label='+', returns='Value')
        # FP - False Positives
        statistics_dictionary['FP'] = false_pos_neg_ensemble(train_dataframe,
                                                             test_dataframe,
                                                             label='+')
        # TN - True Negatives
        statistics_dictionary['TN'] = ensemble_prediction_accuracy(
            train_dataframe, test_dataframe, label='-', returns='Value')
        # FN - False Negatives
        statistics_dictionary['FN'] = false_pos_neg_ensemble(train_dataframe,
                                                             test_dataframe,
                                                             label='-')
        # TPR - True positive rate - TP/(TP+FN)
        statistics_dictionary['TPR'] = round(statistics_dictionary['TP'] / (
                statistics_dictionary['TP'] + statistics_dictionary['FN']), 2)
        # TNR - True negative rate - TN/(TN+FP)
        statistics_dictionary['TNR'] = round(statistics_dictionary['TN'] / (
                statistics_dictionary['TN'] + statistics_dictionary['FP']), 2)
        # accuracy
        statistics_dictionary['Accuracy'] = ensemble_prediction_accuracy(
            train_dataframe, test_dataframe)
    elif method == 'W2':
    # TP - True Positives
        statistics_dictionary['TP'] = w_calc_prediction_accuracy(
            train_dataframe, test_dataframe, 2, label='+', returns='Value')
        # FP - False Positives
        statistics_dictionary['FP'] = false_pos_neg_w(train_dataframe,
                                                      test_dataframe, 2,
                                                      label='+')
        # TN - True Negatives
        statistics_dictionary['TN'] = w_calc_prediction_accuracy(
            train_dataframe, test_dataframe, 2, label='-', returns='Value')
        # FN - False Negatives
        statistics_dictionary['FN'] = false_pos_neg_w(train_dataframe,
                                                      test_dataframe, 2,
                                                      label='-')
        # TPR - True positive rate - TP/(TP+FN)
        statistics_dictionary['TPR'] = round(statistics_dictionary['TP'] / (
                statistics_dictionary['TP'] + statistics_dictionary['FN']), 2)
        # TNR - True negative rate - TN/(TN+FP)
        statistics_dictionary['TNR'] = round(statistics_dictionary['TN'] / (
                statistics_dictionary['TN'] + statistics_dictionary['FP']), 2)
        # accuracy
        statistics_dictionary['Accuracy'] = w_calc_prediction_accuracy(
            train_dataframe, test_dataframe, 2)
    elif method == 'W3':
        # TP - True Positives
        statistics_dictionary['TP'] = w_calc_prediction_accuracy(
            train_dataframe, test_dataframe, 3, label='+', returns='Value')
        # FP - False Positives
        statistics_dictionary['FP'] = false_pos_neg_w(train_dataframe,
                                                      test_dataframe, 3,
                                                      label='+')
        # TN - True Negatives
        statistics_dictionary['TN'] = w_calc_prediction_accuracy(
            train_dataframe, test_dataframe, 3, label='-', returns='Value')
        # FN - False Negatives
        statistics_dictionary['FN'] = false_pos_neg_w(train_dataframe,
                                                      test_dataframe, 3,
                                                      label='-')
        # TPR - True positive rate - TP/(TP+FN)
        statistics_dictionary['TPR'] = round(statistics_dictionary['TP'] / (
                statistics_dictionary['TP'] + statistics_dictionary['FN']), 2)
        # TNR - True negative rate - TN/(TN+FP)
        statistics_dictionary['TNR'] = round(statistics_dictionary['TN'] / (
                statistics_dictionary['TN'] + statistics_dictionary['FP']), 2)
        # accuracy
        statistics_dictionary['Accuracy'] = w_calc_prediction_accuracy(
            train_dataframe, test_dataframe, 3)
    elif method == 'W4':
    # TP - True Positives
        statistics_dictionary['TP'] = w_calc_prediction_accuracy(
            train_dataframe, test_dataframe, 4, label='+', returns='Value')
        # FP - False Positives
        statistics_dictionary['FP'] = false_pos_neg_w(train_dataframe,
                                                      test_dataframe, 4,
                                                      label='+')
        # TN - True Negatives
        statistics_dictionary['TN'] = w_calc_prediction_accuracy(
            train_dataframe, test_dataframe, 4, label='-', returns='Value')
        # FN - False Negatives
        statistics_dictionary['FN'] = false_pos_neg_w(train_dataframe,
                                                      test_dataframe, 4,
                                                      label='-')
        # TPR - True positive rate - TP/(TP+FN)
        statistics_dictionary['TPR'] = round(statistics_dictionary['TP'] / (
                statistics_dictionary['TP'] + statistics_dictionary['FN']), 2)
        # TNR - True negative rate - TN/(TN+FP)
        statistics_dictionary['TNR'] = round(statistics_dictionary['TN'] / (
                statistics_dictionary['TN'] + statistics_dictionary['FP']), 2)
        # accuracy
        statistics_dictionary['Accuracy'] = w_calc_prediction_accuracy(
                                    train_dataframe, test_dataframe, 4)
    else:
        print('Error method input invalid.')

    stats_list = list(statistics_dictionary.values())
    return stats_list

