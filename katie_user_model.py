import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import pickle
import my_functions as fns
from bokeh.plotting import figure, show, output_file
from bokeh.embed import components
from flask import Flask, render_template
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import gridspec
from io import BytesIO
import seaborn as sns
sns.set_context('poster', font_scale=2)

def make_fig(ind):
	selected_feature_names = pd.read_csv('./data/features_rf_smo10featuresnorm_interested.csv')
	selected_feature_names = list(selected_feature_names['0'])
	count_norm1i, count_norm0i, count_norm1e, count_norm0e, count_norm1s, count_norm0s, bins1i, bins0i, bins1e, bins0e, bins1s, bins0s = fns.plot_active_features(selected_feature_names, ind)
	sns.set_context('poster', font_scale=1.3)
	figdict = dict([("w_pad", 3), ("hpad", 1)])
	fig = plt.figure(figsize=(20, 8)) # , subplotpars=figdict 
	ax = [0, 1, 2]
	gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1]) 
	ax[0] = plt.subplot(gs[0])

	iaint = ax[0].bar(bins0i[1:], count_norm0i, alpha=.5, color='r', width=.1, align='center')
	aint = ax[0].bar(bins1i[1:], count_norm1i, alpha=.5, color='b', width=.1, align='center')

	ax[1] = plt.subplot(gs[1])
	iaeng = ax[1].bar(bins0e[1:], count_norm0e, alpha=.5, color='r', width=.1, align='center')
	aeng = ax[1].bar(bins1e[1:], count_norm1e, alpha=.5, color='b', width=.1, align='center')

	ax[2] = plt.subplot(gs[2])
	iasub = ax[2].bar(bins0s[1:], count_norm0s, alpha=.5, color='r', width=.1, align='center')
	asub = ax[2].bar(bins1s[1:], count_norm1s, alpha=.5, color='b', width=.1, align='center')

	ax[0].set_title('Inactive vs. 2+ Days')   # \n \"Interested\" 
	# ax[0].legend((aint, iaint), ('Active', 'Inactive'))
	ax[0].set_xlabel(selected_feature_names[ind])
	ax[0].set_ylabel("% of Patients")

	ax[1].set_title('Inactive vs. 14+ Days')   #  \n \"Engaged\"
	# ax[1].legend((aeng, iaeng), ('Active', 'Inactive'))
	ax[1].set_xlabel(selected_feature_names[ind])
	ax[1].set_ylabel("% of Patients")

	ax[2].set_title('Inactive vs. 30+ Days')   #  \n \"Subscribed\"
	ax[2].legend((asub, iasub), ('Active', 'Inactive'), loc='center left', bbox_to_anchor=(1, 0.5))
	ax[2].set_xlabel(selected_feature_names[ind])
	ax[2].set_ylabel("% of Patients")
	canvas = FigureCanvas(fig)
	img = BytesIO()
	fig.savefig(img)
	img.seek(0)
	return img

def make_figure():
	final_features_raw_wid, final_features_raw, active_all = fns.import_features()
	selected_feature_names = pd.read_csv('./data/features_rf_smo10featuresnorm_interested.csv')
	selected_feature_names = list(selected_feature_names['0'])
	temp = final_features_raw_wid.merge(active_all, on='patient_id')

	activegroup = pd.groupby(temp, by='isactive_interested')
	active = activegroup.get_group(1)
	inactive = activegroup.get_group(0)

	count1, bins1 = np.histogram(active[selected_feature_names[0]])
	count_norm1 = count1 / count1.sum()
	count0, bins0 = np.histogram(inactive[selected_feature_names[0]])
	count_norm0 = count0 / count0.sum()

	fig = plt.figure()
	plt.bar(bins1[1:], count_norm1, alpha=.4, color='r', width=.1, align='center')
	plt.bar(bins0[1:], count_norm0, alpha=.4, color='b', width=.1, align='center')
	plt.title(selected_feature_names[0])
	plt.ylabel('% of Patients')
	plt.legend(['Active', 'Inactive'], loc='upper left' )

	canvas = FigureCanvas(fig)
	img = BytesIO()
	fig.savefig(img)
	img.seek(0)
	return img

def ModelIt():
	# import model
	# with open('/Users/katieporter/Dropbox/Insight/CT/ct_private/rf_smo10featuresnorm_interested.b', 'rb') as f:
	with open('./data/rf_smo10featuresnorm_interested.b', 'rb') as f:
    # load using pickle de-serializer
		deployed_model= pickle.loads(f)
	
	# import complete dataset 
	final_features_raw_wid, final_features_raw, active_all = fns.import_features()

	# get normalizing measures 
	final_features_raw_array = np.array(final_features_raw_wid.drop('patient_id', axis = 1))
	final_features_mean = np.mean(final_features_raw_array, axis=0)
	final_features_std = np.std(final_features_raw_array, axis=0)

	# get the selected features
	# selected_feature_names = pd.read_csv('/Users/katieporter/Dropbox/Insight/CT/ct_private/features_rf_smo10featuresnorm_interested.csv')
	selected_feature_names = pd.read_csv('./data/features_rf_smo10featuresnorm_interested.csv')
	selected_feature_names = list(selected_feature_names['0'])
	
	# import test data and labels 
	test_data, test_labels, trainval_data, trainval_labels = fns.load_train_test_data()
	
	test_noid = test_data.drop('patient_id', axis=1)
	test_features = np.array(test_noid)
	test_feature_norm = (test_features - final_features_mean) / final_features_std 
	test_colnames = list(test_noid.columns.values)

	test_data_norm = pd.DataFrame(test_feature_norm, columns = test_colnames)

	selected_features = pd.DataFrame()
	for i in selected_feature_names:
		selected_features[i] = test_data_norm[i]
	
	# turn dataframes into arrays
	testX = np.array(selected_features)
	testy = np.array(test_labels['isactive_interested'])
	
	pred = deployed_model.predict(testX)
	accuracy = accuracy_score(pred, testy)
	accuracy_perc = accuracy*100

	deployed_model.fit(testX,testy)

	temp = final_features_raw_wid.merge(active_all, on='patient_id')
	mean_diff_interested, mean_diff_engaged, mean_diff_subscribed = fns.get_avg_diff(temp, selected_feature_names)
	mean_diff = mean_diff_interested
	# mean_diff_interested = np.array(mean_diff_interested)
	# mean_diff_interested = list(mean_diff_interested)
	features_out = pd.DataFrame({'feature': selected_feature_names, 'importance': deployed_model.feature_importances_, 'diff2days': mean_diff}) # 'diff_14_days': mean_diff_engaged, 'diff_30_days': mean_diff_subscribed}
	features_out['importance'] = features_out['importance'].round(3)
	features_out['diff2days'] = features_out['diff2days'].round(3)
	# features_out['diff_14_days'] = features_out['diff_14_days'].round(3)
	# features_out['diff_30_days'] = features_out['diff_30_days'].round(3)
	features_out = features_out.sort_values(by='importance', axis=0, ascending=False )
	print(features_out['diff2days'], features_out['importance'])
	return np.round(accuracy_perc, 1), features_out # np.round(accuracy_train, 3),

# def ModelOne(fromUser  = 'Default', patient = []):
def ModelOne(patient):
		# import model
	# with open('/Users/katieporter/Dropbox/Insight/CT/ct_private/rf_smo10featuresnorm_interested.b', 'rb') as f:
	with open('./data/rf_smo10featuresnorm_interested.b', 'rb') as f:

    # load using pickle de-serializer
		deployed_model= pickle.load(f)
	
	# import complete dataset 
	final_features_raw_wid, final_features_raw, active_all = fns.import_features()

	# get normalizing measures 
	final_features_raw_array = np.array(final_features_raw_wid.drop('patient_id', axis = 1))
	final_features_mean = np.mean(final_features_raw_array, axis=0)
	final_features_std = np.std(final_features_raw_array, axis=0)

	# get the selected features
	# selected_feature_names = pd.read_csv('/Users/katieporter/Dropbox/Insight/CT/ct_private/features_rf_smo10featuresnorm_interested.csv')
	selected_feature_names = pd.read_csv('./data/features_rf_smo10featuresnorm_interested.csv')
	selected_feature_names = list(selected_feature_names['0'])

	try:
		patient=int(patient)
		# get row for just this patient
		single_patient = final_features_raw_wid[final_features_raw_wid['patient_id']==patient]
		

		single_patient_noid = single_patient.drop('patient_id', axis=1)
		test_features = np.array(single_patient_noid)
		test_feature_norm = (test_features - final_features_mean) / final_features_std 
		test_colnames = list(single_patient_noid.columns.values)

		test_data_norm = pd.DataFrame(test_feature_norm, columns = test_colnames)


		selected_features = pd.DataFrame()
		for i in selected_feature_names:
			selected_features[i] = test_data_norm[i]


		# extract status of patient (active/inactive)
		temp = active_all[active_all['patient_id']==patient]
		temp = active_all['isactive_interested']
		temp2 = np.array(temp)
		active_status = temp2[0]
		if active_status==1:
			activity = 'is active'
		else:
		    activity = 'is not active'

		# extract first 5 avg acc
		temp = single_patient['mean_reaction_time'][single_patient['mean_reaction_time']>-1]
		temp2 = np.array(temp)
		avg_rt = temp2[0]

		# # extract first 5 avg rt
		temp = single_patient['first_trial_reaction_time'][single_patient['first_trial_reaction_time']>-1]
		temp2 = np.array(temp)
		first_trial_rt = temp2[0]

		# # extract first 5 avg ratio completed
		temp = single_patient['mean_accuracy'][single_patient['mean_accuracy']>-1]
		temp2 = np.array(temp)
		avg_acc = temp2[0]

		# # extract first acc
		temp = single_patient['task_level1_x_avg_accuracy'][single_patient['task_level1_x_avg_accuracy']>-1]
		temp2 = np.array(temp)
		level1acc = temp2[0]

		# extract platform
		temp = single_patient['first_trial_accuracy'][single_patient['first_trial_accuracy']>-1]
		temp2 = np.array(temp)
		first_acc = temp2[0]

		# extract platform
		temp = single_patient['client_platform'][single_patient['client_platform']>-1]
		temp2 = np.array(temp)
		platform= temp2[0]

		# extract platform
		temp = single_patient['task_type37_x_avg_accuracy'][single_patient['task_type37_x_avg_accuracy']>-1]
		temp2 = np.array(temp)
		type37acc= temp2[0]

		temp = single_patient['sum_skipped_trials'][single_patient['sum_skipped_trials']>-1]
		temp2 = np.array(temp)
		sumskipped= temp2[0]

		temp = single_patient['task_level2_x_avg_accuracy'][single_patient['task_level2_x_avg_accuracy']>-1]
		temp2 = np.array(temp)
		level2acc= temp2[0]
		
		temp = single_patient['task_type24_x_avg_accuracy'][single_patient['task_type24_x_avg_accuracy']>-1]
		temp2 = np.array(temp)
		type24acc = temp2[0]	

		testX = np.array(selected_features)
		pred = deployed_model.predict_proba(testX)
		prediction = pred[0][0]
		# prediction = prediction.round(3)
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

		prediction = prediction * 100
		prediction = prediction.round(3)
	except IndexError:
		prediction = 'not calculable'; activity = 'is nonexistent'; assessment = 'please try a different patient id. Hint: try one greater than 1100!'
		patient = '-'; active_status = '-'; avg_rt= '-'; first_trial_rt= '-'; avg_acc= '-'; level1acc= '-'; first_acc= '-'; platform= '-'; type37acc= '-';sumskipped= '-'; level2acc= '-'; type24acc= '-';
	except ValueError:
		prediction = 'not calculable'; activity = 'is nonexistent'; assessment = 'please try a different patient id. Hint: try one greater than 1100!'
		patient = '-'; active_status = '-'; avg_rt= '-'; first_trial_rt= '-'; avg_acc= '-'; level1acc= '-'; first_acc= '-'; platform= '-'; type37acc= '-';sumskipped= '-'; level2acc= '-'; type24acc= '-';
	return prediction, activity, assessment, patient, active_status, avg_rt, first_trial_rt, avg_acc, level1acc, first_acc, platform, type37acc, sumskipped, level2acc, type24acc   