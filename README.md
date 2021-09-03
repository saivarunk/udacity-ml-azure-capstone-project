# Capstone Project - Machine Learning Engineer with Microsoft Azure

- This project has been submitted as part of the Machine Learning Engineer with Microsoft Azure Nanodegree.
- In this project, we have create two models: one using Automated ML (denoted as AutoML from now on) and one customized model whose hyperparameters are tuned using HyperDrive.
- We have used an external dataset instead of one's provided in Azure ML Studio.

## Dataset

### Overview

In this problem, we are using Heart Failure Prediction dataset. from Kaggle. The dataset has the below set of 12 features and a target variable :

| column        | Description  |
| :----- | :---------------|
|age | Age of the patient|
|amaemia| Decrease of red blood cells or hemoglobin|
|creatinine_phosphokinase| Level of the CPK enzyme in the blood (mcg/L)|
|diabetes| If the patient has diabetes|
|ejection_fraction| Percentage of blood leaving the heart at each contraction|
high_blood_pressure| If the patient has hypertension|
|platelets| Platelets in the blood (kiloplatelets/mL)|
|serum_creatinine| Level of serum creatinine in the blood (mg/dL)|
|serum_sodium| Level of serum sodium in the blood (mEq/L)|
|sex| Woman or man|
|smoking| If the patient smokes or not|
|time| Follow-up period (days)|
|DEATH_EVENT (target)| If the patient deceased during the follow-up period (boolean)|

### Task

In this project, we will be using the 12 features to predict DEATH_EVENT outcome, which is If the patient deceased during the follow-up period (boolean).

This is a binary classification problem.

### Access

The dataset has been downloaded from Kaggle and uploaded to this GitHub repository. The dataset is then accessed as a TabularDataset using the URL of the raw .csv file.

```bash
https://raw.githubusercontent.com/saivarunk/udacity-ml-azure-capstone-project/master/dataset/heart_failure_clinical_records_dataset.csv
```

## Automated ML

The AutomatedML Run was created using an instance of AutoMLConfig. The AutoML Config Class is a way of leveraging the AutoML SDK to automate machine learning. 

The following parameters have been used for the Auto ML Run.

```bash
# AutoML settings
automl_settings = {
    "task": "classification",
    "debug_log": 'automl_errors.log',
    "training_data": train_data,
    "label_column_name": 'DEATH_EVENT',
    "compute_target": compute_cluster,
    "enable_early_stopping" : True,
    "experiment_timeout_minutes": 30,
    "n_cross_validations": 4,
    "featurization": 'auto',
    "primary_metric": 'accuracy',
    "verbosity": logging.INFO
}

# AutoML config initialization
automl_config = AutoMLConfig(**automl_settings)
```


| Parameter        | Value          | Description  |
| :----- |:-----:| :---------------|
| task     | 'classification' | Classification is selected since we are performing binary classification |
| label_column_name | 'DEATH_EVENT' | DEATH_EVENT is the target column, hence we are passing it |
| experiment_timeout_minutes | 30  | We can select an optimum time period to complete the automl experiment |
| primary_metric | 'accuracy'    | This metric will be used by AutoML to optimize for model_selection |
| enable_early_stopping | True | Early Stopping is enabled to terminate a run in case the score is not improving in short term. This allows AutoML to train more better models in shorter timeframe |
| featurization | 'auto'   | Featurization is set to auto so that the featurization step is done automatically |
| n_cross_validations | 4  | No. of cross validation splits to be done during the experiment |


### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording

[https://www.youtube.com/watch?v=xhgn7Pc3Ehc](https://www.youtube.com/watch?v=xhgn7Pc3Ehc)
