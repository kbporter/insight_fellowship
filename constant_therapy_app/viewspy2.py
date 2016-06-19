from flask import render_template
from flask import request
from constant_therapy_app import app
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import pandas as pd
import psycopg2
import os 
os.chdir('/Users/katieporter/Dropbox/Insight/CT/ct_share/insight_fellowship/constant_therapy_app')
import sys
sys.path.append('//Users/katieporter/Dropbox/Insight/CT/ct_share/insight_fellowship/constant_therapy_app')

from test import *

from katie_user_model import ModelIt, ModelOne
os.chdir('/Users/katieporter/Dropbox/Insight/CT/ct_share/insight_fellowship')
import numpy as np

user = 'katie' #add your username here (same as previous postgreSQL)            
host = 'localhost'
dbname = 'birth_db'
db = create_engine('postgres://%s%s/%s'%(user,host,dbname))
con = None
con = psycopg2.connect(database = dbname, user = user)

@app.route('/')
@app.route('/index')
def index():
    return render_template("index.html",
       title = 'Home', user = { 'nickname': 'Katie' },
       )

# @app.route('/db')
# def birth_page():
#     sql_query = """                                                             
#                 SELECT * FROM birth_data_table WHERE delivery_method='Cesarean'\
# ;                                                                               
#                 """
#     query_results = pd.read_sql_query(sql_query,con)
#     births = ""
#     print query_results[:10]
#     for i in range(0,10):
#         births += query_results.iloc[i]['birth_month']
#         births += "<br>"
#     return births

# @app.route('/db_fancy')
# def cesareans_page_fancy():
#     sql_query = """
#                SELECT index, attendant, birth_month FROM birth_data_table WHERE delivery_method='Cesarean';
#                 """
#     query_results=pd.read_sql_query(sql_query,con)
#     births = []
#     for i in range(0,query_results.shape[0]):
#         births.append(dict(index=query_results.iloc[i]['index'], attendant=query_results.iloc[i]['attendant'], birth_month=query_results.iloc[i]['birth_month']))
#     return render_template('cesareans.html',births=births)

@app.route('/input')
def ct_input():
    return render_template("input.html")

@app.route('/features_output')
def features_output():
	#pull 'birth_month' from input field and store it
    numfeatures = request.args.get('numfeatures')
    try: 
        numfeatures = int(numfeatures)
    except ValueError:
        numfeatures = 5
    validation, table = ModelIt() # training,
    table_out = []
    # for i in range(0, table.shape[0]):
    for i in range(numfeatures):
        table_out.append(dict(feature=table.iloc[i]['feature'], importance=table.iloc[i]['importance']))

    return render_template("features_output.html", table=table_out, validation=validation, numfeatures=numfeatures) #training=training,

@app.route('/patient_output')
def patient_output():
    #pull 'patient_id' from input field and store it
    patient = request.args.get('idofpatient')
    # patient = int(patient)
    prediction, activity, assessment, patient, avg_acc, avg_rt, avg_ratio, first_acc, platform, active_status = ModelOne(patient)
    return render_template("patient_output.html", prediction=prediction, activity=activity, assessment=assessment, patient=patient, avg_acc=avg_acc, avg_rt=avg_rt, avg_ratio=avg_ratio, first_acc=first_acc, platform=platform, active_status=active_status)

@app.route('/high_risk_output')
def high_risk_output():
    #pull 'patient_id' from input field and store it
    patient = request.args.get('idofpatient')
    # patient = int(patient)
    prediction, activity, assessment, patient, avg_acc, avg_rt, avg_ratio, first_acc, platform, active_status = ModelOne(patient)
    return render_template("high_risk_output.html", prediction=prediction, activity=activity, assessment=assessment, patient=patient, avg_acc=avg_acc, avg_rt=avg_rt, avg_ratio=avg_ratio, first_acc=first_acc, platform=platform, active_status=active_status)




