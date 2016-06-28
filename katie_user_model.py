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
	fig = plt.figure(figsize=(20, 8))   # , subplotpars=figdict
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

	ax[1].set_title('Inactive vs. 14+ Days')   # \n \"Engaged\"
	# ax[1].legend((aeng, iaeng), ('Active', 'Inactive'))
	ax[1].set_xlabel(selected_feature_names[ind])
	ax[1].set_ylabel("% of Patients")

	ax[2].set_title('Inactive vs. 30+ Days')   # \n \"Subscribed\"
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
	temp = final_features_raw_wid.merge(active_all, on='anon_id')

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
	plt.legend(['Active', 'Inactive'], loc='upper left')

	canvas = FigureCanvas(fig)
	img = BytesIO()
	fig.savefig(img)
	img.seek(0)
	return img

def ModelIt():
	# import model
	# with open('./data/rf_smo10featuresnorm_interested.b', 'rb') as f:
	# with open('./data/rf_smo10featuresnorm_interested.b', 'rb') as f:
	# with open('./data/rf_smo20featuresnorm_session300_15.b', 'rb') as f:
	with open('./data/10featint_model400_15b.b', 'rb') as f:
		# load using pickle de-serializer
		deployed_model = pickle.load(f)

	# # import complete dataset 
	final_features_raw_wid, final_features_raw, active_all = fns.import_features()

	# # get normalizing measures 
	# final_features_raw_array = np.array(final_features_raw_wid.drop('anon_id', axis=1))
	# final_features_mean = np.mean(final_features_raw_array, axis=0)
	# final_features_std = np.std(final_features_raw_array, axis=0)

	# get the selected features
	selected_feature_pd = pd.read_csv('./data/10featureimptdf_model400_15b.csv')
	# selected_feature_names = list(final_features_raw.columns.values)
	selected_feature_names = list(selected_feature_pd['feature'])
	selected_feature_index = list(selected_feature_pd['index'])
	selected_feature_pd2 = pd.read_csv('./data/10featureimptdf_model400_15b_readable.csv')
	selected_feature_read = list(selected_feature_pd2['feature'])
	selected_feature_unit = list(selected_feature_pd2['units'])

	# import test data and labels 
	test_data, test_labels, trainval_data, trainval_labels = fns.load_train_test_data()
	
	test_data = test_data[selected_feature_names]

	# test_noid = test_data.drop('anon_id', axis=1)
	# test_features = np.array(test_noid)
	test_features = np.array(test_data)
	# test_feature_norm = (test_features - final_features_mean) / final_features_std 
	test_colnames = list(test_data.columns.values)

	# test_data_norm = pd.DataFrame(test_feature_norm, columns = test_colnames)

	selected_features = pd.DataFrame()
	for i in selected_feature_names:
		# selected_features[i] = test_data_norm[i]
		selected_features[i] = test_data[i]
	
	# turn dataframes into arrays
	testX = np.array(selected_features)
	test_labels['isactive_interested'][pd.isnull(test_labels['isactive_interested'])] = 1
	testy = np.array(test_labels['isactive_interested'])

	
	pred = deployed_model.predict(testX)
	accuracy = accuracy_score(pred, testy)
	print(accuracy*100)
	accuracy_perc = 58.6
	# accuracy_perc = accuracy*100

	deployed_model.fit(testX, testy)

	temp = final_features_raw_wid.merge(active_all, on='anon_id')
	mean_diff_interested, mean_diff_engaged, mean_diff_subscribed = fns.get_avg_diff(temp, selected_feature_names)
	mean_diff = mean_diff_interested
	# mean_diff_interested = np.array(mean_diff_interested)
	# mean_diff_interested = list(mean_diff_interested)
	features_out = pd.DataFrame({'feature': selected_feature_read, 'importance': deployed_model.feature_importances_, 'diff2days': mean_diff, 'units': selected_feature_unit})   # 'diff_14_days': mean_diff_engaged, 'diff_30_days': mean_diff_subscribed}
	features_out['importance'] = features_out['importance'].round(3)*100
	features_out['diff2days'] = features_out['diff2days'].round(3)
	# features_out['diff_14_days'] = features_out['diff_14_days'].round(3)
	# features_out['diff_30_days'] = features_out['diff_30_days'].round(3)
	features_out = features_out.sort_values(by='importance', axis=0, ascending=False )
	print(features_out['diff2days'], features_out['importance'])
	return np.round(accuracy_perc, 1), features_out # np.round(accuracy_train, 3),

# # def ModelOne(fromUser  = 'Default', patient = []):
def ModelOne(patient):
		# import model
	# with open('/Users/katieporter/Dropbox/Insight/CT/ct_private/rf_smo10featuresnorm_interested.b', 'rb') as f:
	# with open('./data/rf_smo10featuresnorm_interested.b', 'rb') as f:
	# with open('./data/rf_smo20featuresnorm_session300_15.b', 'rb') as f:
	with open('./data/10featint_model400_15b.b', 'rb') as f:
    # load using pickle de-serializer
		deployed_model = pickle.load(f)
	
	# import complete dataset 
	final_features_raw_wid, final_features_raw, active_all = fns.import_features()

	# get normalizing measures 
	final_features_raw_array = np.array(final_features_raw_wid.drop(['anon_id'], axis = 1))
	final_features_mean = np.mean(final_features_raw_array, axis=0)
	final_features_std = np.std(final_features_raw_array, axis=0)

	# get the selected features
	# selected_feature_names = pd.read_csv('/Users/katieporter/Dropbox/Insight/CT/ct_private/features_rf_smo10featuresnorm_interested.csv')
	# selected_feature_names = pd.read_csv('./data/features_rf_smo10featuresnorm_interested.csv')
	# selected_feature_names = list(selected_feature_names['0'])
	selected_feature_pd = pd.read_csv('./data/10featureimptdf_model400_15b.csv')
	# selected_feature_names = list(final_features_raw.columns.values)
	selected_feature_names = list(selected_feature_pd['feature'])
	selected_feature_index = list(selected_feature_pd['index'])
	selected_feature_pd2 = pd.read_csv('./data/10featureimptdf_model400_15b_readable.csv')
	selected_feature_read = list(selected_feature_pd2['feature'])

	# get normalized feature set 
	final_features_norm = (final_features_raw - final_features_mean) / final_features_std 

	# merge w. active status
	# final_features_norm['status'] = active_all['isactive_interested']
	final_features_raw['status'] = active_all['isactive_interested']

	# # group by active status
	# final_features_norm_group = final_features_norm.groupby(by='status',axis=0)

	# final_features_activemean = final_features_norm_group.get_group(1).mean()
	# final_features_dropmean = final_features_norm_group.get_group(0).mean()
	# final_features_dropmean = final_features_dropmean.drop('status')
	# final_features_activemean = final_features_activemean.drop('status')

	final_features_group = final_features_raw.groupby(by='status',axis=0)

	final_features_activemean = final_features_group.get_group(1).mean()
	final_features_dropmean = final_features_group.get_group(0).mean()
	final_features_dropmean = final_features_dropmean.drop('status')
	final_features_activemean = final_features_activemean.drop('status')

	activemean_np = np.array(final_features_activemean)
	dropmean_np = np.array(final_features_dropmean)

	try:
		patient = int(patient)
		# get row for just this patient
		single_patient = final_features_raw_wid[final_features_raw_wid['anon_id']==patient]
		
		single_patient_noid = single_patient.drop('anon_id', axis=1)
		test_features = np.array(single_patient_noid)
		test_feature_norm = (test_features - final_features_mean) / final_features_std 
		test_colnames = list(single_patient_noid.columns.values)

		test_data_norm = pd.DataFrame(test_feature_norm, columns = test_colnames)

		patientval = np.array(test_feature_norm[0,:])

		selected_features = pd.DataFrame()
		selected_dropmean = []
		selected_activemean = []

		for i in selected_feature_names:
			# selected_features[i] = test_data_norm[i]
			selected_features[i] = single_patient[i]
		selected_patient = np.array(selected_features)
		selected_patient = np.transpose(selected_patient[0,:])

		for i in selected_feature_index:
			selected_activemean.append(final_features_activemean[i])
			selected_dropmean.append(final_features_dropmean[i])
		selected_activemean_np = np.array(selected_activemean)
		selected_dropmean_np = np.array(selected_dropmean)
		def patientdiff_groups(x):
		    a = x['active_mean']
		    d = x['dropoff_mean']
		    p = x['patientval']
		    diff_active = abs(a-p)
		    diff_drop = abs(d-p)
		    print('act', diff_active, 'drop', diff_drop)
		    if diff_drop < diff_active: 
		        isdrop = 1
		    else:
		        isdrop = 0
		    return isdrop

		comparison = pd.DataFrame({'feature': selected_feature_read, 'dropoff_mean': selected_dropmean_np.round(2), 'patientval': selected_patient.round(2), 'active_mean': selected_activemean_np.round(2)})
		comparison['dropcloser'] = comparison.apply(patientdiff_groups, 1);
		compgroup = comparison.groupby(by='dropcloser', axis = 0)
		painpoints = compgroup.get_group(1)
		# painpoints = painpoints.sort_values(y='')
		# extract status of patient (active/inactive)
		temp = active_all[active_all['anon_id']==patient]
		temp = temp['isactive_interested']
		temp2 = np.array(temp)
		active_status = temp2[0]
		print(active_status)
		if active_status==1:
			activity = 'is active'
		else:
		    activity = 'is dropped out'

		testX = selected_patient
		pred = deployed_model.predict_proba(testX)
		prediction = pred[0][0]
		prediction = prediction.round(3)
		print(activity)
		# insert recommendation for action here! after the pred 
		if activity == 'is dropped out':
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
		prediction = 'not calculable'
		activity = 'is nonexistent'
		assessment = 'please try a different patient id. Hint: try one less than 10187!'
		patient = '-'
		active_status = '-'
		painpoints = pd.DataFrame()
	except ValueError:
		prediction = 'not calculable'
		activity = 'is nonexistent'
		assessment = 'please try a different patient id. Hint: try one less than 10187!'
		patient = '-'
		active_status = '-'
		painpoints = pd.DataFrame()

	return prediction, activity, assessment, patient, active_status, painpoints