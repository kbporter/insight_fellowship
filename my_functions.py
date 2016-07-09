import time
import pandas as pd
import numpy as np


def maketime_sec(x):
    """ returns time in seconds from timestamp """
    new1 = time.strptime(x, "%Y-%m-%d %H:%M:%S")
    new2 = time.mktime(new1)
    return new2


def isactive_interested(x):
    """ determines whether user is active based on # days since last activity,
        cutoff is 1 day. Input is # days """
    if x > 1:
        x = 1
    else:
        x = 0
    return x


def isactive_engaged(x):
    """ determines whether user is active based on # days since last activity,
        cutoff is 2 weeks. Input is # days """
    if x > 14:
        x = 1
    else:
        x = 0
    return x


def isactive_subscribed(x):
    """ determines whether user is active based on # days since last activity,
        cutoff is 30 days. Input is # days """
    if x > 30:
        x = 1
    else:
        x = 0
    return x


def import_features():
    """ imports the features and labels (active user definitions) """
    # load features
    final_features_raw_wid = pd.read_csv('./data/anon_features_wid.csv')
    # drop the id for using in model
    final_features_raw = final_features_raw_wid.drop('anon_id', axis=1)
    # load labels
    active_nonan = pd.read_csv('./data/active_anon.csv')
    # clean irrelevant columns make new label structure
    final_features_raw = final_features_raw.drop('Unnamed: 0', axis=1)
    final_features_raw_wid = final_features_raw_wid.drop('Unnamed: 0', axis=1)
    active_all = pd.DataFrame({'anon_id': active_nonan['anon_id'],
                               'isactive_interested': active_nonan['isactive_interested'],
                               'isactive_engaged': active_nonan['isactive_engaged'],
                               'isactive_subscribed': active_nonan['isactive_subscribed'],
                               'greater1sess': active_nonan['greater1sess']
                               })

    return final_features_raw_wid, final_features_raw, active_all


def load_train_test_data():
    """ for developing model, load train / test data separately """

    test_data = pd.read_csv('./data/anon_int_testdata.csv')
    test_data = test_data.drop('Unnamed: 0', axis=1)

    test_labels = pd.read_csv('./data/anon_int_testlabels.csv')
    test_labels = test_labels.drop('Unnamed: 0', axis=1)

    trainval_data = pd.read_csv('./data/anon_int_traindata.csv')
    trainval_data = trainval_data.drop('Unnamed: 0', axis=1)

    trainval_labels = pd.read_csv('./data/anon_int_trainlabels.csv')
    trainval_labels = trainval_labels.drop('Unnamed: 0', axis=1)

    return test_data, test_labels, trainval_data, trainval_labels


def get_avg_diff(temp, selected_feature_names):
    """ get the average difference between active / drop-off groups,
        for each active label definition.
        temp = dataframe of features with labels,
        selected_feature_names = list of names of features to be included """

    # grouping by label, for each definition of 'active'
    active_interested_group = pd.groupby(temp, by='isactive_interested')
    active_interested = active_interested_group.get_group(1)
    inactive_interested = active_interested_group.get_group(0)

    active_engaged_group = pd.groupby(temp, by='isactive_engaged')
    active_engaged = active_engaged_group.get_group(1)
    inactive_engaged = active_engaged_group.get_group(0)

    active_subscribed_group = pd.groupby(temp, by='isactive_subscribed')
    active_subscribed = active_subscribed_group.get_group(1)
    inactive_subscribed = active_subscribed_group.get_group(0)

    # extract the difference between group averages for features included in model
    mean_diff_interested = []
    mean_diff_engaged = []
    mean_diff_subscribed = []
    for i in selected_feature_names:
        mean_diff_interested.append(active_interested[i].mean() - inactive_interested[i].mean())
        mean_diff_engaged.append(active_engaged[i].mean() - inactive_engaged[i].mean())
        mean_diff_subscribed.append(active_subscribed[i].mean() - inactive_subscribed[i].mean())

    return mean_diff_interested, mean_diff_engaged, mean_diff_subscribed


def patientdiff_groups(x):
    """ identify features where this patient is more similar to
        drop-off mean than active group mean.
        Input = dataframe with group means and single patient values """
    a = x['active_mean']
    d = x['dropoff_mean']
    p = x['patientval']
    diff_active = abs(a - p)
    diff_drop = abs(d - p)
    # print('act', diff_active, 'drop', diff_drop)
    # if distance to drop-off group is closer than active group,
    # mark to include (1)
    if diff_drop < diff_active:
        isdrop = 1
    else:
        isdrop = 0
    return isdrop
