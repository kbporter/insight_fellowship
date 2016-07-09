/* NOTES
-don't want to add columns to a table, make new tables
-try not to have queries within queries
-the responses table is optimized for patient_id, so better to query one patient at a time from python */

-- get the target patient users who are only using the app at home, not with clinician
CREATE TEMPORARY TABLE constant_therapy.tmp_patientid 
SELECT 
    DISTINCT patient_id 
FROM 
    constant_therapy.sessions 
WHERE 
    patient_id > 1000 
        AND task_type_id IS NOT NULL 
        AND completed_task_count > 0
        AND patient_id NOT IN (SELECT 
            DISTINCT patient_id 
            FROM 
                constant_therapy.sessions 
            WHERE 
                type = 'ASSISTED');

-- get os platform for each target user
SELECT 
    id, client_platform
FROM
    constant_therapy.users
WHERE
    id IN (SELECT 
            *
        FROM
            constant_therapy.tmp_patientid);

-- count # not null values for platform 
SELECT 
    COUNT(client_platform)
FROM
    constant_therapy.users
WHERE
    id IN (SELECT 
            *
        FROM
            constant_therapy.tmp_patientid)
        AND client_platform IS NOT NULL;


-- get first / last activity timestamps
SELECT 
    patient_id, MIN(start_time), MAX(start_time)
FROM
    sessions
WHERE
    patient_id IN (SELECT 
            *
        FROM
            tmp_patientid)
GROUP BY patient_id;

-- make table so can access first/last accuracy more easily
create temporary table first_last_activity2 SELECT 
    patient_id, MIN(start_time) as min_start, MAX(start_time) as max_start
FROM
    sessions
WHERE
    patient_id IN (SELECT 
            *
        FROM
            tmp_patientid)
GROUP BY patient_id;

-- get accuracy for first / last session completed      
SELECT 
    sessions.patient_id, sessions.accuracy, sessions.start_time
FROM
    first_last_activity2
        LEFT JOIN
    sessions ON first_last_activity2.min_start = sessions.start_time
WHERE
    accuracy IS NOT NULL
LIMIT 10; -- ASC for first task DESC for last task


-- get the lead source for patients who only have scheduled sessions 
SELECT 
    id, lead_source 
FROM 
    ct_customer.customers 
WHERE 
    id IN (SELECT 
            * 
        FROM 
            tmp_patientid);

-- get the number of sessions completed (looking for > 1 vs. 1 to define active/inactive) 
SELECT 
    patient_id, COUNT(distinct id) AS num_sessions 
FROM 
    constant_therapy.sessions 
WHERE 
    patient_id IN (SELECT 
            * 
        FROM 
            tmp_patientid) 
GROUP BY patient_id;
