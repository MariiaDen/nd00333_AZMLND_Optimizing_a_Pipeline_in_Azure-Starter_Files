# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
The given dataset is bankmarketing_train.csv. This is a sample dataset provided by Microsoft. According to their documentation, the y column indicates if a customer subscribed to a fixed term deposit. The dataset contains information about customers - their job, marital status, education, loan, etc.
This is a classification problem, where we tried to predict whether a customer is subscribed to a fixed-term deposit, or not. 

I have tried out to analyze the cleaned, as well as the raw data. After trying out different models, the best accuracy reached for different approaches was:

| Data   | Approach | Accuracy |
| ------------- | ---------|------------- |
| Cleaned data | SKLearn - Logistic Regression  | 0.91181  |
| Original data | AutoML - VotingEnsemble  | 0.91697  |
| Cleaned data | AutoML - VotingEnsemble  | 0.91643  |

As it can be seen, all three approaches have found best models with similar results. 

## Introduction
The aim of this project was to create the following pipeline: 

![Alt text](screenshots/0.%20creating-and-optimizing-an-ml-pipeline.png?raw=true "Optional Title")

As can be seen at the first glance, the whole project consists of three major elements: 
- train.py
- Jupyter notebook
- Research report

Research report is the document you are reading now. The other two had to be uploaded to the Azure environment, to the Notebook first: 

![Alt text](screenshots/1.%20Uploaded%20files.PNG?raw=true "Optional Title")

In order to make the scripts work, a compute instance is needed. For that, a Standard_D2_v2 machine was selected:

![Alt text](screenshots/2.%20Creating%20Compute.PNG?raw=true "Optional Title")

This instance was named as "Udacity-Lab1". 

## train.py
The whole ML pipeline begins with the train.py script, which is pre-given. The main aim of this script is data preparation. There, the dataset had to be uploaded, cleaned and split into training and testing set. In order to read the data, the TabularDatasetFactory was used. 

```
datastore_path = "https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv"
ds = TabularDatasetFactory.from_delimited_files(path=datastore_path)
```

The Logistic Regression accepts two arguments: regularization strength and maximum number of iterations to converge. Per default, they are equal to 1.0 and 100 respectively. According to the definition, logistic regression is a statistical model that in its basic form uses a logistic function to model a binary dependent variable. Mathematically, a binary logistic model has a dependent variable with two possible values, such as "0" and "1".

<img src="https://miro.medium.com/max/640/0*gKOV65tvGfY8SMem.png" height="200" />

The train.py is then called from the Jupyter notebook. 

## Jupyter notebook
### New experiment
The Jupyter notebook has several logical parts. At first, the new experiment must be created. The one here was named as "quick-starts-experiment". 

![Alt text](screenshots/4.%20Workspace%20found.PNG?raw=true "Optional Title")

After that, it was checked whether compute instance exists. The official Microsoft documentation was consulted here [Link](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-create-attach-compute-cluster?tabs=python). To create a persistent Azure Machine Learning Compute resource in Python, the vm_size and max_nodes properties were specified accordingly (STANDARD_D2_V2 and 4 nodes).

### Hyper Drive
After that, the preparations for experiment begin. Hyperparameters are adjustable parameters that let you control the model training process. Hyperparameter tuning is the process of finding the configuration of hyperparameters that results in the best performance. Azure ML automates this process. 
First of all, there must have been defined two parameters, which need to be sent to train.py - regularization strength and maximum number of iterations. The RandomParameterSampling allows different combinations of both parameters when on the run. The regularization strength lies between 0.001 and 1 - the lower the value, the bigger impact. The maximal iterations - between 30 and 250. After having several issues with uniform, the choice function was selected to define hyperparameters as discrete. More about different functions can be read here: [Link](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.parameter_expressions?view=azure-ml-py)

```
ps = RandomParameterSampling(
    {
        "--C": choice(0.001, 0.01, 0.1, 1.0),
        "--max_iter": choice(30, 50, 100, 250)
    }
)
```
Random sampling supports discrete and continuous hyperparameters. It supports early termination of low-performance runs. Accuracy was selected as the primary metric, which is looked at to get the highest value. Apart of that, the early termination policy was defined. This helps to save time by stopping runs in case of poor performance. There are four different termination policies to choose from:
- Bandit policy
- Median stopping policy
- Truncation selection policy
- No termination policy

In this work, the Bandit policy was selected. Bandit terminates runs where the primary metric is not within the specified slack factor/slack amount compared to the best performing run. After configurations were done, the Hyper Drive could be submitted for a run: 

![Alt text](screenshots/8.%20RunDetails.PNG?raw=true "Optional Title")

After the experiment has finished, the best run could be defined as follows:
![Alt text](screenshots/11.%20Best%20Run%20ID%20and%20Accuracy.PNG?raw=true "Optional Title")


### AutoML
AutoML though took some more time to execute, it worked in a different way and showed a slightly better results. It can be used as alternative, since requires less programming skills. It trains and tunes a model using the target metric that is specified. Since less programming is needed, it saves some time and resources. 

For AutoML two runs were done - with raw data, by only using the dataset as it is, and also with the cleaned data. 
In the first case, the following data was used:

```
datastore_path = "https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv"
ds = TabularDatasetFactory.from_delimited_files(path=datastore_path)
```
Pay attention, that ds variable is ussed as training data:
```
automl_config = AutoMLConfig(
    experiment_timeout_minutes=30,
    task="classification",
    primary_metric="accuracy",
    training_data=ds,
    label_column_name="y",
    n_cross_validations=5,
    compute_target=cpu_cluster)
```

In the second try, the cleaned data was used. For that it was first turned into DataFrame by pandas, and then after all manipulations, it was transformed back to the Dataset, as the only data format that can be accepted as training data in AutoMLConfig. 

The best run had an ID AutoML_a892b78e-55a3-4bbe-9978-22ac4cfadd88_24, and can be seen below:

![Alt text](screenshots/18.%20Run%20Complete%20with%20Cleanup_3.PNG?raw=true "Optional Title")

The AutoML has defined the following top fetures by their importance:
- duration
- nr.employed
- emp.variate

These are the key predictors that model uses to generate results.

After redoing the lab, by skipping Jupyter Notebook and generating directly the AutoML work, the results turned out to be similar. 
The best model is still the same, and has following most important features: 
![Alt text](screenshots/Best%20run.PNG?raw=true "Optional Title")

![Alt text](screenshots/Best%20individual%20feature.PNG?raw=true "Optional Title")

![Alt text](screenshots/Summary%20importance.PNG?raw=true "Optional Title")

By selecting an option "Explain model" we can also generate results for other models. Let's check the third best:

![Alt text](screenshots/Third%20best.PNG?raw=true "Optional Title")

After generating results, we come up with the same most important features:

![Alt text](screenshots/XGBoost.PNG?raw=true "Optional Title")

## Pipeline comparison
As previously mentioned, the programming part in AutoML is much shorter, therefore saves time. However, compute time took much longer - about 40 minutes, which is about four times longer than the first approach with HyperDrive. At the same time, if there is any error in python code or in parateters configurations, the Hyper Drive case might need to be re-run several times, which again takes compute time. Therefore, when going for one of two options, these factors need to be taken into considerations. 

The accuracy of both approaches was practically equally good. This might be due to the high number of iterations selected for the logictic regression - up to 250. And as results have shown, the best result has worked with 250 iterations. 

## Future work
As a part of the future work, the different hyperparameters might be reviewed and re-thinked. For example, the random sampling is often used as initial search, and then refined to improve results. Therefore, the random sampling could be the first thing to start with.

## Proof of cluster clean up
After the work is done, the cluster should be cleaned up. It can be deleted with the following command: 
```
cpu_cluster.delete()
```
Right after this code is executed, the changes can be seen:

![Alt text](screenshots/19.%20Deleting.PNG?raw=true "Optional Title")

After the compute instance is deleted, nothing is listed:

![Alt text](screenshots/20.%20Deleted.PNG?raw=true "Optional Title")
