# insight_fellowship

This is *partial* code for a website and model to predict drop-off completed during the Insight Data Science Fellowship. Since this was a consultation project for a patient focused app (Constant Therapy) not all of the code/data is included. 

GOAL:
- predict user drop-off & identify important features predictive of drop-off
- more information at www.churnnomore.me

Tools Used: 
- MySQL
- Python
- Flask
- Bootstrap

SQL queries: 
- cleanqueries.sql (exctraction of data/features in MySQLWorkbench) 
- sql_queries.py (extraction of data/features in python)

Python Functions:
- my_functions.py (functions used in data processing)
- katie_user_model.py (functions used by app to run model and output result) 

Web App:
constant_therapy_app/
  - views.py (renders each page for the app, and runs models in katie_user_model.py)
  - templates (contains html templates for each page of churnnomore.me) 
  - static (contains images, css, javascript for app) 
