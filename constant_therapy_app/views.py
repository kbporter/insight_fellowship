from flask import render_template
from flask import request, send_file
from constant_therapy_app import app
import numpy as np
from katie_user_model import *
import sys
# sys.path.append('/Users/katieporter/Dropbox/Insight/CT/ct_share/insight_fellowship/constant_therapy_app')
sys.path.append('/home/ubuntu/from_github/constant_therapy_app')


@app.route('/')
@app.route('/input')
def ct_input():
    return render_template("input.html")


@app.route('/index')
def index():
    return render_template("index.html",
                           title='Home',
                           user={'nickname': 'Katie'})


@app.route('/features_output')
def features_output():
    """ output feature importance and model performance """

    # get number of features to display
    numfeatures = request.args.get('numfeatures')
    # print(numfeatures)

    # default to top 5 if out of range
    try:
        numfeatures = int(numfeatures)
        if numfeatures == 0:
            numfeatures = 5
        elif numfeatures > 10:
            numfeatures = 10
        elif numfeatures < 0:
            numfeatures = 5
    except ValueError:
        numfeatures = 5

    # run the model
    validation, table = ModelIt()

    # get table with # features specified above,
    # as a dict so will print on html page
    table_out = []
    for i in range(numfeatures):
        table_out.append(dict(feature=table.iloc[i]['feature'],
                              importance=table.iloc[i]['importance'],
                              diff2days=table.iloc[i]['diff2days'],
                              units=table.iloc[i]['units']))

    return render_template("features_output.html", table=table_out,
                           validation=validation,
                           numfeatures=numfeatures)


@app.route('/patient_output')
def patient_output():
    """ run model for single patient """
    # pull 'anon_id' from input field and store it
    patient = request.args.get('idofpatient')
    # print('patient', patient)

    # default to patient id = 100 if out of range
    try:
        if int(patient) < 0:
            patient = 100
        elif int(patient) == 0:
            patient = 100
    except ValueError:
        patient = 100

    # run model on single patient
    prediction, activity, assessment, patient, active_status, comparison = ModelOne(patient)

    # turn painpoint features into dict for html table
    table_out = []
    for i in np.arange(comparison.shape[0]):
        table_out.append(dict(feature=comparison.iloc[i]['feature'],
                              dropoff_mean=comparison.iloc[i]['dropoff_mean'],
                              patientval=comparison.iloc[i]['patientval'],
                              activemean=comparison.iloc[i]['active_mean']))

    return render_template("patient_output.html",
                           prediction=prediction, activity=activity,
                           assessment=assessment, patient=patient,
                           active_status=active_status, comptable=table_out)


@app.route('/patient_output_present')
def patient_output_present():
    """ duplicate of patient_output page with larger font for presentation """

    # pull 'anon_id' from input field and store it
    patient = request.args.get('idofpatient')
    # print('patient', patient)

    # if patient is out of range default to 100
    try:
        if int(patient) < 0:
            patient = 100
        elif int(patient) == 0:
            patient = 100
    except ValueError:
        patient = 100

    # run model on single patient
    prediction, activity, assessment, patient, active_status, comparison = ModelOne(patient)

    # turn painpoint features into dict for html table
    table_out = []
    for i in np.arange(comparison.shape[0]):
        table_out.append(dict(feature=comparison.iloc[i]['feature'],
                              dropoff_mean=comparison.iloc[i]['dropoff_mean'],
                              patientval=comparison.iloc[i]['patientval'],
                              activemean=comparison.iloc[i]['active_mean']))

    return render_template("patient_output_present.html",
                           prediction=prediction, activity=activity,
                           assessment=assessment, patient=patient,
                           active_status=active_status, comptable=table_out)


@app.route('/slides')
def slides():
    return render_template("slides.html")


@app.route('/aboutme')
def aboutme():
    return render_template("aboutme.html")


@app.route('/contact')
def contact():
    return render_template("contact.html")
