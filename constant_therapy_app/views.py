from flask import render_template
from flask import request, send_file
from constant_therapy_app import app
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import pandas as pd
import psycopg2
import os 
from bokeh.embed import components
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from io import BytesIO

import sys
# sys.path.append('/Users/katieporter/Dropbox/Insight/CT/ct_share/insight_fellowship/constant_therapy_app')
sys.path.append('/home/ubuntu/from_github/constant_therapy_app')

# from test import *

from katie_user_model import *
# os.chdir('/Users/katieporter/Dropbox/Insight/CT/ct_share/insight_fellowship')
import numpy as np

# user = 'katie' #add your username here (same as previous postgreSQL)            
# host = 'localhost'
# dbname = 'birth_db'
# db = create_engine('postgres://%s%s/%s'%(user,host,dbname))
# con = None
# con = psycopg2.connect(database = dbname, user = user)

@app.route('/')
@app.route('/input')
def ct_input():
    return render_template("input.html")

@app.route('/index')
def index():
    return render_template("index.html",
       title = 'Home', user = { 'nickname': 'Katie' },
       )

@app.route('/figure1')
def figure1():
    numfeatures = request.args.get('numfeatures')
    img = make_fig(0)
#     script, div = components(plot)
    return send_file(img, mimetype='image/png')

@app.route('/figure2')
def figure2():
    img = make_fig(1)
    return send_file(img, mimetype='image/png')

@app.route('/figure3')
def figure3():
    img = make_fig(2)
    return send_file(img, mimetype='image/png')

@app.route('/figure4')
def figure4():
    img = make_fig(3)
    return send_file(img, mimetype='image/png')

@app.route('/figure5')
def figure5():
    img = make_fig(4)
    return send_file(img, mimetype='image/png')

@app.route('/figure6')
def figure6():
    img = make_fig(5)
    return send_file(img, mimetype='image/png')

@app.route('/figure7')
def figure7():
    img = make_fig(6)
    return send_file(img, mimetype='image/png')

@app.route('/figure8')
def figure8():
    img = make_fig(7)
    return send_file(img, mimetype='image/png')

@app.route('/figure9')
def figure9():
    img = make_fig(8)
    return send_file(img, mimetype='image/png')

@app.route('/figure10')
def figure10():
    img = make_fig(9)
    return send_file(img, mimetype='image/png')

@app.route('/features_output')
def features_output():
	#pull 'birth_month' from input field and store it
    numfeatures = request.args.get('numfeatures')
    print(numfeatures)
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
    print(numfeatures)
    validation, table = ModelIt() # training,
    table_out = []
    # for i in range(0, table.shape[0]):
    for i in range(numfeatures):
        table_out.append(dict(feature=table.iloc[i]['feature'], importance=table.iloc[i]['importance'], diff2days=table.iloc[i]['diff2days']))

    return render_template("features_output.html", table=table_out, validation=validation, numfeatures=numfeatures) #training=training,

@app.route('/patient_output')
def patient_output():
    #pull 'anon_id' from input field and store it
    patient = request.args.get('idofpatient')
    print('patient', patient)
    try: 
        if int(patient) < 0:
            patient = 100
        elif int(patient) ==0:
            patient = 100 
    except ValueError:
        patient = 100

    # patient = int(patient)
    prediction, activity, assessment, patient, active_status, comparison  = ModelOne(patient) #  avg_rt, first_trial_rt, avg_acc, level1acc, first_acc, platform, type37acc, sumskipped, level2acc, type24acc,
    table_out = []
    for i in np.arange(comparison.shape[0]):
        table_out.append(dict(feature=comparison.iloc[i]['feature'], dropoff_mean=comparison.iloc[i]['dropoff_mean'], patientval=comparison.iloc[i]['patientval'], activemean=comparison.iloc[i]['active_mean'])) # 

    return render_template("patient_output.html", prediction=prediction, activity=activity, assessment=assessment, patient=patient, active_status=active_status, comptable=table_out) #avg_rt = avg_rt, first_trial_rt = first_trial_rt, avg_acc = avg_acc, level1acc=level1acc, first_acc=first_acc, platform=platform, type37acc=type37acc, sumskipped=sumskipped, level2acc=level2acc, type24acc=type24acc, 

@app.route('/patient_output_present')
def patient_output_present():
    #pull 'anon_id' from input field and store it
    patient = request.args.get('idofpatient')
    print('patient', patient)
    try: 
        if int(patient) < 0:
            patient = 100
        elif int(patient) ==0:
            patient = 100 
    except ValueError:
        patient = 100

    # patient = int(patient)
    prediction, activity, assessment, patient, active_status, comparison  = ModelOne(patient) #  avg_rt, first_trial_rt, avg_acc, level1acc, first_acc, platform, type37acc, sumskipped, level2acc, type24acc,
    table_out = []
    for i in np.arange(comparison.shape[0]):
        table_out.append(dict(feature=comparison.iloc[i]['feature'], dropoff_mean=comparison.iloc[i]['dropoff_mean'], patientval=comparison.iloc[i]['patientval'], activemean=comparison.iloc[i]['active_mean'])) # 

    return render_template("patient_output_present.html", prediction=prediction, activity=activity, assessment=assessment, patient=patient, active_status=active_status, comptable=table_out) #avg_rt = avg_rt, first_trial_rt = first_trial_rt, avg_acc = avg_acc, level1acc=level1acc, first_acc=first_acc, platform=platform, type37acc=type37acc, sumskipped=sumskipped, level2acc=level2acc, type24acc=type24acc, 


@app.route('/slides')
def slides():
    return render_template("slides.html")

@app.route('/aboutme')
def aboutme():
    return render_template("aboutme.html")

@app.route('/contact')
def contact():
    return render_template("contact.html")



