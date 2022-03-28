
# def predicting_labels(train_dataframe, test_dataframe):
#     """Computes the predicted labels of a given test set of data based on
#     probabilities determined on training set of data. Calculates this based
#     on how many of the previous consecutive positive or negative days precede
#     the day on which we are predicting. Sign is assigned based on the
#     probability determined from test set of a negative or positive based on
#     these consecutive days. """
#     true_labels = test_dataframe['True Label'].values
#     predicted_signs = []
#     for index, value in enumerate(true_labels):
#         if true_labels[index-1] != value:
#             if default_probability_pos(train_dataframe) > default_probability_neg(train_dataframe):
#                 predicted_signs.append('+')
#             else:
#                 predicted_signs.append('-')
#         elif true_labels[index-1] == value:
#             days = 1
#             if true_labels[index-2] == value:
#                 days = 2
#                 if true_labels[index-3] == value:
#                     days = 3
#                     if consecutive_days(train_dataframe, days, '+') > \
#                             consecutive_days(train_dataframe, days, '-'):
#                         predicted_signs.append('+')
#
#                     else:
#                         predicted_signs.append('-')
#
#                 else:
#                     if consecutive_days(train_dataframe, days, '+') > \
#                             consecutive_days(train_dataframe, days, '-'):
#                         predicted_signs.append('+')
#
#                     else:
#                         predicted_signs.append('-')
#
#             else:
#                 if consecutive_days(train_dataframe, days, '+') > consecutive_days(
#                         train_dataframe, days, '-'):
#                     predicted_signs.append('+')
#
#                 else:
#                     predicted_signs.append('-')
#
#         else:
#             if default_probability_pos(train_dataframe) > default_probability_neg(
#                     train_dataframe):
#                 predicted_signs.append('+')
#
#             else:
#                 predicted_signs.append('-')
#
#     return predicted_signs
#

# def consecutive_days_list(test_dataframe):
#     """Returns a list containing the number of consecutive days (positive or
#     negative) for the test dataframe set. Used to calculate the accuracy of
#     our predicting models."""
#     true_labels = test_dataframe['True Label'].values
#     day_list = []
#     days = 0
#     for index, value in enumerate(true_labels):
#         # Had to force the first index to return a day of 0 as it was giving
#         # 2 for some reason
#         if index == 0:
#             days = 0
#             day_list.append(days)
#         else:
#             if true_labels[index-1] != value:
#                 days = 0
#                 day_list.append(days)
#             elif true_labels[index-1] == value:
#                 days = 1
#                 if true_labels[index-2] == value:
#                     days = 2
#                     if true_labels[index-3] == value:
#                         days = 3
#                         day_list.append(days)
#                     else:
#                         day_list.append(days)
#                 else:
#                     day_list.append(days)
#             else:
#                 day_list.append(days)
#
#     return day_list
#
