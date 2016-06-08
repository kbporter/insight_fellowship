-- NOTES -- 
-- don't want to add columns to a table, make new tables --
-- try not to have queries within queries -- 
-- the responses table is optimized for patient_id, so better to query one patient at a time from python --

-- get the target patient users who are only using the app at home, not with clinician -- 
create temporary table constant_therapy.tmp_patientid SELECT distinct patient_id from constant_therapy.sessions where 
patient_id > 1000 AND patient_id NOT IN (SELECT distinct patient_id 
from constant_therapy.sessions where type = 'ASSISTED');

-- get os platform for each target user -- 
SELECT 
    id, client_platform
FROM
    constant_therapy.users
WHERE
    id IN (SELECT 
            *
        FROM
            constant_therapy.tmp_patientid);

-- count # not null values for platform -- 
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


-- get first / last activity timestamps -- 
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

-- make table so can access first/last accuracy more easily --
create temporary table first_last_activity SELECT 
    patient_id, MIN(start_time) as min_start, MAX(start_time) as max_start
FROM
    sessions
WHERE
    patient_id IN (SELECT 
            *
        FROM
            tmp_patientid)
GROUP BY patient_id;

-- get accuracy for first / last session completed --        
SELECT 
    patient_id, accuracy, start_time
FROM
    first_last_activity
LEFT JOIN sessions
ON first_last_activity.MIN(start_time) = sessions.start_time; -- ASC for first task DESC for last task --


select patient_id, accuracy, start_time from sessions where patient_id = 48514 order by start_time asc;

