import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def ModelIt(patient):
	my_features_pd = pd.read_csv('/Users/katieporter/Dropbox/Insight/CT/ct_private/MVP_features.csv')
	my_labels_pd = pd.read_csv('/Users/katieporter/Dropbox/Insight/CT/ct_private/MVP_labels.csv')

	my_features = np.array(my_features_pd)
	my_features = my_features[:, 1:]
	my_labels = np.array(my_labels_pd['is_active'])
	feature_names = list(my_features_pd.columns.values)
	feature_names = feature_names[1:]
	features_norm = (my_features - np.mean(my_features, axis=0)) / np.std(my_features, axis=0)

	# getting 50% of the data for training
	X_train, X_valtest, y_train, y_valtest = train_test_split(features_norm, my_labels, test_size=0.5, random_state=42)

	# splitting the remaining 50% for validation / testing 
	X_validate, X_test, y_validate, y_test = train_test_split(X_valtest, y_valtest, test_size=0.5, random_state=42)
	my_features_pd = pd.read_csv('/Users/katieporter/Dropbox/Insight/CT/ct_private/MVP_features.csv')

	# Random Forest
	model2 = RandomForestClassifier()
	model2.fit(X_train, y_train)
	features_out = pd.DataFrame({'feature': feature_names, 'importance': model2.feature_importances_})

	# print features_out

	# no feature selection
	model_simple = LogisticRegression()
	model_simple.fit(X_train, y_train)
	
	# get the prediction + accuracy for validation data
	pred = model_simple.predict(X_validate)
	accuracy_validate = accuracy_score(pred, y_validate)
	
	# get the prediction + accuracy for training data
	pred_train = model_simple.predict(X_train)
	accuracy_train = accuracy_score(pred_train, y_train)
	
	# print 'training accuracy is', accuracy_train, ', validation accuracy is', accuracy1
	# print(accuracy1, feature_names)

	return accuracy_train, accuracy_validate, features_out
