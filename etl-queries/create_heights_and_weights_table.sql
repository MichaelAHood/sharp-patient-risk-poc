CREATE TABLE height_weight AS

(
SELECT 
      encntr_id
    , hw
    , AVG(value) as avg_value
    
FROM

(
SELECT  
      encntr_id
    , CASE WHEN event_title_text='Height/Length' THEN 'height' ELSE 'weight' END AS hw
    , CAST(result_val as FLOAT) AS value
    , display

FROM clinical_event ce
LEFT JOIN
         
         (
          SELECT code_value, display
          FROM code_value cv
          WHERE cv.code_value = '246'
             OR cv.code_value = '271'
          ) d

ON ce.result_units_cd = d.code_value
WHERE event_cd = '2700653'
OR event_cd = '4674677'

) a
 
GROUP BY encntr_id, hw 
)