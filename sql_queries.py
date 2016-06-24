#Queries SQL for the age groups of the database
from time import sleep
import pandas as pd
import time

def last_5_sessions(cur, patient):
	""" for a single patient, get duration, accuracy, latency, task level """
	""" and completed task ratio for the 5 most recent sessions """
	cur.execute(("select patient_id, duration, accuracy, latency, task_level, (completed_task_count/total_task_count) as completed_ratio  from sessions where patient_id = ({patient_num}) order by start_time desc limit 5;").format(patient_num=patient))
	sleep(.5)
	result = cur.fetchall()
	# r = [i[0] for i in result]
	r = pd.DataFrame(result,columns=['patient_id', 'duration', 'accuracy', 'latency','task_level', 'completed_ratio'])
	return r

def maketime_sec(x):
    """ returns time in seconds from timestamp """
    new1 = time.strptime(x,"%Y-%m-%d %H:%M:%S")
    new2 = time.mktime(new1)
    return new2


def first_5_sessions(cur, patient):
	""" for a single patient, get duration, accuracy, latency, task level """
	""" and completed task ratio for the 5 most recent sessions """
	cur.execute(("select patient_id, duration, accuracy, latency, task_level, (completed_task_count/total_task_count) as completed_ratio  from sessions where patient_id = ({patient_num}) order by start_time asc limit 5;").format(patient_num=patient))
	sleep(.5)
	result = cur.fetchall()
	# r = [i[0] for i in result]
	r = pd.DataFrame(result,columns=['patient_id', 'duration', 'accuracy', 'latency','task_level', 'completed_ratio'])
	return r


def responses_first_session(cur, patient):
	""" for a single patient, get all response info from first session """
	# get first session id number
	cur.execute(("select min(id) from sessions where patient_id = {patient_num} and task_type_id is not null;").format(patient_num=patient))
	sleep(.5)
	sess_id = cur.fetchall()
	sleep(.5)
	# get responses from that first session
	cur.execute(("select session_id, patient_id, accuracy, latency, start_time, end_time, duration, task_type, task_json, additional_data_json, system_version_number, comment, skipped, task_level, task_type_id, client_os_version, client_hardware_type FROM constant_therapy.responses where session_id = {sess_num};").format(sess_num=sess_id[0][0]))
	sleep(.5)
	result = cur.fetchall()
	# r = [i[0] for i in result]
	r = pd.DataFrame(result, columns=['session_id', 'patient_id', 'accuracy', 'latency', 'start_time', 'end_time', 'duration', 'task_type', 'task_json', 'additional_data_json', 'system_version_number', 'comment', 'skipped', 'task_level', 'task_type_id', 'client_os_version', 'client_hardware_type'])
	return r

def patient_deficit(cur, patient):
	""" for a single patient, get their deficit type """ 
	cur.execute(("select * from ct_customer.customers_to_deficits where customer_id = {patient_num};").format(patient_num=patient))
	sleep(.5)
	result = cur.fetchall()
	r = pd.DataFrame(result, columns=['customer_id', 'deficit_id', 'details'])
	return r

def patient_disorder(cur, patient):
	""" for a single patient, get their disorder type """ 
	cur.execute(("select * from ct_customer.customers_to_disorders where customer_id = {patient_num};").format(patient_num=patient))
	sleep(.5)
	result = cur.fetchall()
	r = pd.DataFrame(result, columns=['customer_id', 'disorder_id', 'details'])
	return r

def last_activity(cur, patient):
	""" for a single patient, get their last activity timestamp """ 
	cur.execute(("select session_id, patient_id, max(start_time) as max_start FROM constant_therapy.responses where patient_id = {patient_num} and accuracy is not null;").format(patient_num=patient))
	sleep(.5)
	result = cur.fetchall()
	r = pd.DataFrame(result, columns=['session_id', 'patient_id', 'max_start'])
	return r

def max_difficulty(cur):
	""" get the highest task level (most difficult) for each task type """
	cur.execute(("select task_type_id, max(task_level) as highest_task_level from task_progression group by task_type_id;"))
	sleep(.5)
	result = cur.fetchall()
	r = pd.DataFrame(result, columns=['task_type_id', 'highest_task_level'])
	return r

def lead_source(cur):
	cur.execute("create temporary table constant_therapy.tmp_patientid SELECT distinct patient_id from constant_therapy.sessions where patient_id > 1000 AND patient_id NOT IN (SELECT distinct patient_id from constant_therapy.sessions where type = 'ASSISTED');")
	sleep(.5)
	cur.execute("SELECT id, lead_source FROM ct_customer.customers WHERE id IN (SELECT * FROM tmp_patientid);")
	result = cur.fetchall()
	r = pd.DataFrame(result, columns=['patient_id', 'lead_source'])
	return r

def num_sessions(cur):
	cur.execute("create temporary table constant_therapy.tmp_patientid SELECT distinct patient_id from constant_therapy.sessions where patient_id > 1000 AND patient_id NOT IN (SELECT distinct patient_id from constant_therapy.sessions where type = 'ASSISTED');")
	sleep(.5)
	cur.execute("SELECT patient_id, COUNT(distinct id) as num_sessions FROM constant_therapy.sessions WHERE patient_id IN (SELECT * FROM tmp_patientid) group by patient_id;")
	result = cur.fetchall()
	r = pd.DataFrame(result, columns=['patient_id', 'num_sessions'])
	return r
