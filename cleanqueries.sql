-- don't want to add columns to a table, make new tables --
-- try not to have queries within queries -- 
-- the responses table is optimized for patient_id, so better to query one patient at a time from python --

create temporary table constant_therapy.tmp_patientid SELECT distinct patient_id from constant_therapy.sessions where 
patient_id > 1000 AND patient_id NOT IN (SELECT distinct patient_id 
from constant_therapy.sessions where type = 'ASSISTED');

