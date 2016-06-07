#Queries SQL for the age groups of the database
from time import sleep
import pandas as pd
import time

# cur = cnx.cursor();

# def scheduled_patients(cur):
# 	""" select patients who are not developers (id >1000) and who only have scheduled at home sessions """
# 	scheduled_patients = '''SELECT count(distinct patient_id) from constant_therapy.sessions where patient_id = 1039 AND patient_id NOT IN (SELECT distinct patient_id from constant_therapy.sessions where type = 'ASSISTED'); '''
# 	cur.execute(scheduled_patients)
# 	sleep(.5)
# 	result = cur.fetchall()
# 	r = [i[0] for i in result]

# 	#r = pd.DataFrame(result,columns=['task_id', 'avg_acc', 'num_items', 'num_skipped','skip_ratio']
# 	return r    

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
