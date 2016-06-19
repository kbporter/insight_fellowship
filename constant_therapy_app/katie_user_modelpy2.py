import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import pickle

def ModelIt():
	# import model
	with open('/Users/katieporter/Dropbox/Insight/CT/ct_private/random_forest_model.b', 'r') as f:
    # load using pickle de-serializer
		deployed_model= pickle.load(f)
	# import complete dataset 
	final_features_raw = pd.read_csv('/Users/katieporter/Dropbox/Insight/CT/ct_private/final_all_features_noid.csv')
	final_features_raw = final_features_raw.drop('Unnamed: 0', axis=1)

	# get list of feature names 
	feature_names = list(final_features_raw.columns.values)


	# import test data and labels 
	testX_pd = pd.read_csv('/Users/katieporter/Dropbox/Insight/CT/ct_private/test_data.csv')
	testy_pd = pd.read_csv('/Users/katieporter/Dropbox/Insight/CT/ct_private/test_label.csv')
	# drop duplicate index column 
	testX_pd = testX_pd.drop('Unnamed: 0', axis=1)
	testy_pd = testy_pd.drop('Unnamed: 0', axis=1)
	
	# turn dataframes into arrays
	testX = np.array(testX_pd)
	testy = np.array(testy_pd)
	
	pred = deployed_model.predict(testX)
	accuracy = accuracy_score(pred, testy)
	accuracy_perc = accuracy*100

	deployed_model.fit(testX,testy)

	
	features_out = pd.DataFrame({'feature': feature_names, 'importance': deployed_model.feature_importances_})
	features_out = features_out.round(3)
	features_out = features_out.sort(columns='importance', axis=0, ascending=False )
	
	return np.round(accuracy_perc, 1), features_out # np.round(accuracy_train, 3),

# def ModelOne(fromUser  = 'Default', patient = []):
def ModelOne(patient):
	with open('/Users/katieporter/Dropbox/Insight/CT/ct_share/insight_fellowship/constant_therapy_app/model_balanced.b', 'r') as f:
    # load using pickle de-serializer
		deployed_model= pickle.load(f)
	# deployed_model = pickle.load('model.b', 'rb')
	
	my_features_pd = pd.read_csv('/Users/katieporter/Dropbox/Insight/CT/ct_private/MVP_features.csv')
	my_features = np.array(my_features_pd)
	my_features = my_features[:, 1:]

	feature_key = pd.read_csv('/Users/katieporter/Dropbox/Insight/CT/ct_private/all_data_noNan.csv')
	try:
		patient=int(patient)
		# get row for just this patient
		single_patient = feature_key[feature_key['patients']==patient]
		
		# extract status of patient (active/inactive)
		temp = single_patient['is_active'][single_patient['is_active']>-1]
		temp2 = np.array(temp)
		active_status = temp2[0]
		if active_status==1:
			activity = 'is active'
		else:
		    activity = 'is not active'

		# extract first 5 avg acc
		temp = single_patient['avg_accuracy'][single_patient['avg_accuracy']>-1]
		temp2 = np.array(temp)
		avg_acc = temp2[0]

		# extract first 5 avg rt
		temp = single_patient['avg_latency'][single_patient['avg_latency']>-1]
		temp2 = np.array(temp)
		avg_rt = temp2[0]

		# extract first 5 avg ratio completed
		temp = single_patient['avg_ratio'][single_patient['avg_ratio']>-1]
		temp2 = np.array(temp)
		avg_ratio = temp2[0]

		# extract first acc
		temp = single_patient['first_acc'][single_patient['first_acc']>-1]
		temp2 = np.array(temp)
		first_acc = temp2[0]

		# extract platform
		temp = single_patient['platform'][single_patient['platform']>-1]
		temp2 = np.array(temp)
		platform = temp2[0]

		# load feature names 
		my_features_pd = pd.read_csv('/Users/katieporter/Dropbox/Insight/CT/ct_private/MVP_features.csv')
		feature_names = list(my_features_pd.columns.values)
		feature_names = feature_names[1:]
		
		single_features = []
		for i in feature_names:
			# value = single_patient.get_value(1, i, takeable=False)
			tempval = single_patient[i][~np.isnan(single_patient[i])]
			tempval2 = np.array(tempval)
			single_features.append(tempval2[0])

		single_features = np.array(single_features)
		pred = deployed_model.predict_proba(single_features)
		prediction = pred[0][0]
		prediction = prediction.round(3)
		# insert recommendation for action here! after the pred 
		if activity == 'is not active':
			if prediction > .5: 
				assessment = 'The model predicted this user correctly'
			elif prediction < .5:
				assessment = 'The model was not correct for this user. No one is perfect!'
			elif prediction == .5: 
				assessment = 'This user seems to be on the fence!'
			else:
				assessment = 'Error assessing model accuracy for this user'
		elif activity == 'is active':
			if prediction < .5: 
				assessment = 'The model predicted this user correctly'
			elif prediction > .5:
				assessment = 'The model was not correct for this user. No one is perfect!'
			elif prediction == .5: 
				assessment = 'This user seems to be on the fence!'
			else:
				assessment = 'Error comparing model prediction and activity status for this patient'
		else:
			assessment = 'Error identifying patient activity status'
	except IndexError:
		prediction = 'not calculable'; activity = 'is nonexistent'; assessment = 'please try a different patient id. Hint: try one greater than 1100!'
		patient = '-'; avg_acc='-'; avg_rt='-'; avg_ratio='-'; first_acc='-'; platform='-'; active_status='-'
	except ValueError:
		prediction = 'not calculable'; activity = 'is nonexistent'; assessment = 'please try a different patient id. Hint: try one greater than 1100!'
		patient = '-'; avg_acc='-'; avg_rt='-'; avg_ratio='-'; first_acc='-'; platform='-'; active_status='-'
	return prediction, activity, assessment, patient, avg_acc, avg_rt, avg_ratio, first_acc, platform, active_status