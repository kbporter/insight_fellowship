import time
import pandas as pd
import numpy as np

def maketime_sec(x):
    """ returns time in seconds from timestamp """
    new1 = time.strptime(x,"%Y-%m-%d %H:%M:%S")
    new2 = time.mktime(new1)
    return new2

def isactive_interested(x):
	""" determines whether user is active based on # days since last activity, cutoff is 3 weeks """
	if x > 1:
		x = 1
	else:
		x = 0
	return x

def isactive_engaged(x):
	""" determines whether user is active based on # days since last activity, cutoff is 3 weeks """
	if x > 14:
		x = 1
	else:
		x = 0
	return x

def isactive_subscribed(x):
	""" determines whether user is active based on # days since last activity, cutoff is 3 weeks """
	if x > 30:
		x = 1
	else:
		x = 0
	return x

def import_features():
	# final_features_raw_wid = pd.read_csv('/Users/katieporter/Dropbox/Insight/CT/ct_private/final_all_features_wid.csv')
	# final_features_raw_wid = pd.read_csv('./data/final_all_features_wid.csv')
	final_features_raw_wid = pd.read_csv('./data/anon_features_wid.csv')
	# final_features_raw_wid = pd.read_csv('./data/final_session_20.csv')
	# final_features_raw_wid = final_features_raw_wid.drop('greater1sess', axis=1)

	final_features_raw = final_features_raw_wid.drop('anon_id', axis=1)
	# active_nonan = pd.read_csv('/Users/katieporter/Dropbox/Insight/CT/ct_private/active_nonan.csv')
	active_nonan = pd.read_csv('./data/active_anon.csv')

	final_features_raw = final_features_raw.drop('Unnamed: 0', axis=1)
	final_features_raw_wid = final_features_raw_wid.drop('Unnamed: 0', axis=1)
	active_all = pd.DataFrame({'anon_id': active_nonan['anon_id'], 
                          'isactive_interested': active_nonan['isactive_interested'],
                          'isactive_engaged': active_nonan['isactive_engaged'], 
                          'isactive_subscribed': active_nonan['isactive_subscribed'],
                          'greater1sess': active_nonan['greater1sess']
                          })
	return final_features_raw_wid, final_features_raw, active_all

def import_features_subscribed():		
	final_features_raw_wid = pd.read_csv('./data/features86_smo200_10.csv')
	active_nonan = pd.read_csv('./data/active_anon.csv')

	final_features_raw_wid = final_features_raw_wid.drop('Unnamed: 0', axis=1)
	active_all = pd.DataFrame({'anon_id': active_nonan['anon_id'], 
                          'isactive_interested': active_nonan['isactive_interested'],
                          'isactive_engaged': active_nonan['isactive_engaged'], 
                          'isactive_subscribed': active_nonan['isactive_subscribed'],
                          'greater1sess': active_nonan['greater1sess']})
	final_features_noid = final_features_raw_wid.drop('anon_id', axis=1)
	feature_names = list(final_features_noid.columns.values)

	return final_features_raw_wid, active_all, feature_names

def load_train_test_data(): 
	# test_data = pd.read_csv('./data/anontest_data_final.csv')
	test_data = pd.read_csv('./data/anon_int_testdata.csv')
	test_data = test_data.drop('Unnamed: 0', axis=1)

	# test_labels = pd.read_csv('./data/anontest_labels_final.csv')
	test_labels = pd.read_csv('./data/anon_int_testlabels.csv')
	test_labels = test_labels.drop('Unnamed: 0', axis=1)

	trainval_data= pd.read_csv('./data/anon_int_traindata.csv')
	trainval_data = trainval_data.drop('Unnamed: 0', axis=1)

	trainval_labels = pd.read_csv('./data/anon_int_trainlabels.csv')
	trainval_labels = trainval_labels.drop('Unnamed: 0', axis=1)

	return test_data, test_labels, trainval_data, trainval_labels


def plot_active_features(selected_feature_names, index):
	# split the data into active/inactive 
	final_features_raw_wid, final_features_raw, active_all = import_features()
	temp = final_features_raw_wid.merge(active_all, on='anon_id')
	active_interested_group = pd.groupby(temp, by='isactive_interested')
	active_interested = active_interested_group.get_group(1)
	inactive_interested = active_interested_group.get_group(0)

	count1i, bins1i = np.histogram(active_interested[selected_feature_names[index]])
	count_norm1i = count1i / count1i.sum()
	count0i, bins0i = np.histogram(inactive_interested[selected_feature_names[index]])
	count_norm0i = count0i / count0i.sum()

 
	active_engaged_group = pd.groupby(temp, by='isactive_engaged')
	active_engaged = active_engaged_group.get_group(1)
	inactive_engaged = active_engaged_group.get_group(0)

	count1e, bins1e = np.histogram(active_engaged[selected_feature_names[index]])
	count_norm1e = count1e / count1e.sum()
	count0e, bins0e = np.histogram(inactive_engaged[selected_feature_names[index]])
	count_norm0e = count0e / count0e.sum()


	active_subscribed_group = pd.groupby(temp, by='isactive_subscribed')
	active_subscribed = active_subscribed_group.get_group(1)
	inactive_subscribed = active_subscribed_group.get_group(0)

	count1s, bins1s = np.histogram(active_subscribed[selected_feature_names[index]])
	count_norm1s = count1s / count1s.sum()
	count0s, bins0s = np.histogram(inactive_subscribed[selected_feature_names[index]])
	count_norm0s = count0s / count0s.sum()

	return count_norm1i, count_norm0i, count_norm1e, count_norm0e, count_norm1s, count_norm0s, bins1i, bins0i, bins1e, bins0e, bins1s, bins0s 

def get_avg_diff(temp, selected_feature_names):
	active_interested_group = pd.groupby(temp, by='isactive_interested')
	active_interested = active_interested_group.get_group(1)
	inactive_interested = active_interested_group.get_group(0)

	active_engaged_group = pd.groupby(temp, by='isactive_engaged')
	active_engaged = active_engaged_group.get_group(1)
	inactive_engaged = active_engaged_group.get_group(0)

	active_subscribed_group = pd.groupby(temp, by='isactive_subscribed')
	active_subscribed = active_subscribed_group.get_group(1)
	inactive_subscribed = active_subscribed_group.get_group(0)

	mean_diff_interested = []
	mean_diff_engaged = []
	mean_diff_subscribed = []
	for i in selected_feature_names: 
	    mean_diff_interested.append(active_interested[i].mean() - inactive_interested[i].mean())
	    mean_diff_engaged.append(active_engaged[i].mean() - inactive_engaged[i].mean())
	    mean_diff_subscribed.append(active_subscribed[i].mean() - inactive_subscribed[i].mean())
	return mean_diff_interested, mean_diff_engaged, mean_diff_subscribed
