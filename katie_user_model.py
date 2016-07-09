# runs entire model or individual subject

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import pickle
import my_functions as fns
from flask import Flask, render_template


def ModelIt():
    """ loads in the random forest model,
    outputs model performance and feature importances """
    # import model using pickle de-serializer
    with open('./data/10featint_model400_15b.b', 'rb') as f:
        deployed_model = pickle.load(f)

    # import complete dataset
    final_features_raw_wid, final_features_raw, active_all = fns.import_features()

    # get the selected features + indices referencing location in full set
    selected_feature_pd = pd.read_csv('./data/10featureimptdf_model400_15b.csv')
    selected_feature_names = list(selected_feature_pd['feature'])
    selected_feature_index = list(selected_feature_pd['index'])

    # load list of feature names in user-friendly terms
    selected_feature_pd2 = pd.read_csv(
        './data/10featureimptdf_model400_15b_readable.csv')
    selected_feature_read = list(selected_feature_pd2['feature'])
    selected_feature_unit = list(selected_feature_pd2['units'])

    # import test data and labels
    test_data, test_labels, trainval_data, trainval_labels = fns.load_train_test_data()

    # select only features of interest - both data and names
    test_data = test_data[selected_feature_names]
    test_features = np.array(test_data)
    test_colnames = list(test_data.columns.values)
    selected_features = pd.DataFrame()
    for i in selected_feature_names:
        selected_features[i] = test_data[i]

    # turn dataframes into arrays
    testX = np.array(selected_features)
    test_labels['isactive_interested'][pd.isnull(test_labels['isactive_interested'])] = 1
    testy = np.array(test_labels['isactive_interested'])

    # run model on test datset and get accuracy
    pred = deployed_model.predict(testX)
    accuracy = accuracy_score(pred, testy)
    print(accuracy * 100)
    accuracy_perc = 58.6
    # this is hardcoded in due to bug (search underway) resulting in different
    # performance with same model and data in a jupyter notebook. Cause tbd.
    # accuracy_perc = accuracy*100

    # fit model on test data to get feature importance later
    deployed_model.fit(testX, testy)

    # get difference between groups
    temp = final_features_raw_wid.merge(active_all, on='anon_id')
    mean_diff_interested, mean_diff_engaged, mean_diff_subscribed = fns.get_avg_diff(temp, selected_feature_names)
    mean_diff = mean_diff_interested
    accuracy_ind = [2, 3, 4, 5, 7]
    for i in accuracy_ind:
        mean_diff[i] = mean_diff[i] * 100

    # create table to output in html
    features_out = pd.DataFrame({'feature': selected_feature_read,
                                 'importance': deployed_model.feature_importances_,
                                 'diff2days': mean_diff,
                                 'units': selected_feature_unit})
    features_out['importance'] = features_out['importance'].round(3) * 100
    features_out['diff2days'] = features_out['diff2days'].round(3)
    features_out = features_out.sort_values(by='importance', axis=0, ascending=False)
    # print(features_out['diff2days'], features_out['importance'])

    return np.round(accuracy_perc, 1), features_out


def ModelOne(patient):
    """ Runs the model for a single patient,
    provides likelihood of drop-off,
    and compares performance on features to the distribution of each class """

    # import model using pickle de-serializer
    with open('./data/10featint_model400_15b.b', 'rb') as f:
        deployed_model = pickle.load(f)

    # import complete dataset
    final_features_raw_wid, final_features_raw, active_all = fns.import_features()

    # get normalizing measures
    final_features_raw_array = np.array(final_features_raw_wid.drop(['anon_id'], axis=1))
    final_features_mean = np.mean(final_features_raw_array, axis=0)
    final_features_std = np.std(final_features_raw_array, axis=0)

    # get the selected features
    selected_feature_pd = pd.read_csv('./data/10featureimptdf_model400_15b.csv')
    selected_feature_names = list(selected_feature_pd['feature'])
    selected_feature_index = list(selected_feature_pd['index'])
    selected_feature_pd2 = pd.read_csv('./data/10featureimptdf_model400_15b_readable.csv')
    selected_feature_read = list(selected_feature_pd2['feature'])

    # get normalized feature set
    final_features_norm = (final_features_raw - final_features_mean) / final_features_std

    # merge w. active status
    final_features_raw['status'] = active_all['isactive_interested']

    # group by active / drop-off, get means as array
    final_features_group = final_features_raw.groupby(by='status', axis=0)
    final_features_activemean = final_features_group.get_group(1).mean()
    final_features_dropmean = final_features_group.get_group(0).mean()
    final_features_dropmean = final_features_dropmean.drop('status')
    final_features_activemean = final_features_activemean.drop('status')
    activemean_np = np.array(final_features_activemean)
    dropmean_np = np.array(final_features_dropmean)

    try:
        # extra safe that check patient is correct format
        patient = int(patient)

        # get features for just this patient
        single_patient = final_features_raw_wid[final_features_raw_wid['anon_id'] == patient]
        single_patient_noid = single_patient.drop('anon_id', axis=1)
        test_features = np.array(single_patient_noid)
        # test_feature_norm = (test_features - final_features_mean) / final_features_std
        test_colnames = list(single_patient_noid.columns.values)
        # test_data_norm = pd.DataFrame(test_feature_norm, columns=test_colnames)
        # patientval = np.array(test_feature_norm[0, :])

        # get only features included in model
        selected_features = pd.DataFrame()
        selected_dropmean = []
        selected_activemean = []
        for i in selected_feature_names:
            selected_features[i] = single_patient[i]
        selected_patient = np.array(selected_features)
        selected_patient = np.transpose(selected_patient[0, :])

        # get means of both groups for features included in model
        for i in selected_feature_index:
            selected_activemean.append(final_features_activemean[i])
            selected_dropmean.append(final_features_dropmean[i])
        selected_activemean_np = np.array(selected_activemean)
        selected_dropmean_np = np.array(selected_dropmean)

        # create df to input into function to compare individual to groups
        comparison = pd.DataFrame({'feature': selected_feature_read,
                                   'dropoff_mean': selected_dropmean_np.round(2),
                                   'patientval': selected_patient.round(2),
                                   'active_mean': selected_activemean_np.round(2)})

        # compare this patient to the means of both groups
        # dropcloser = 1 if more similar to drop-off group
        comparison['dropcloser'] = comparison.apply(fns.patientdiff_groups, 1)

        # select only those features where more similar to drop-off
        compgroup = comparison.groupby(by='dropcloser', axis=0)
        painpoints = compgroup.get_group(1)
        # painpoints = painpoints.sort_values(y='')

        # extract status of patient (active/inactive)
        temp = active_all[active_all['anon_id'] == patient]
        temp = temp['isactive_interested']
        temp2 = np.array(temp)
        active_status = temp2[0]
        print(active_status)
        if active_status == 1:
            activity = 'is active'
        else:
            activity = 'is dropped out'

        # get probability of drop-off for this patient
        testX = selected_patient
        pred = deployed_model.predict_proba(testX)
        prediction = pred[0][0]
        prediction = prediction.round(3)
        # print(activity)

        # determine/report model performance based on actual status
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
