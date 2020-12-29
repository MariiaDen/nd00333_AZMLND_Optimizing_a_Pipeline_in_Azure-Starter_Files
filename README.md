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

#### AutoML


**What are the benefits of the parameter sampler you chose?**

**What are the benefits of the early stopping policy you chose?**

## AutoML
**In 1-2 sentences, describe the model and hyperparameters generated by AutoML.**

## Pipeline comparison
**Compare the two models and their performance. What are the differences in accuracy? In architecture? If there was a difference, why do you think there was one?**

## Future work
**What are some areas of improvement for future experiments? Why might these improvements help the model?**
Random sampling supports discrete and continuous hyperparameters. It supports early termination of low-performance runs. Some users do an initial search with random sampling and then refine the search space to improve results.

In random sampling, hyperparameter values are randomly selected from the defined search space.


## Proof of cluster clean up
**If you did not delete your compute cluster in the code, please complete this section. Otherwise, delete this section.**
**Image of cluster marked for deletion**
