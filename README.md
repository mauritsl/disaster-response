# Disaster Response Pipeline Project

The Disaster Response pipeline is a Sklearn / Flask application that analyses text messages in emergency situations.
This is a machine learning / NLP project that uses pre-categorized messages for training, from which it can infer categories
for other messages.

This application is split in three parts:

* ETL-pipeline for extracting and cleaning data from CSV-files
* ML-pipeline that builds a model to categorize messages
* Dashboard that displays information on the testing set and can process individual messages

## Usage

Two CSV-files are required to run this application. These contains the messages and categories.
Sample data is included in the "data" directory.

To run the ETL-pipeline, execute

```
python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
```

Run the ML-piipeline (requires database from last step):

```
python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
```

Run the dashboard:

```
cd app
python run.py
```

Dashboard is available at http://localhost:3001/

## Files

The following files are included in this project:

```
app                             Flask Dashboard
 /run.py                        Application code
 /templates                     HTML templates
   /master.html
   /go.html
models                          ML-pipeline
 /train_classifier.py           Application code
 /classifier.pkl                Export of pre-trained model
data
 /process_data.py               Application code
 /disaster_messages.csv         Sample data
 /disaster_categories.csv       Smample data
```
