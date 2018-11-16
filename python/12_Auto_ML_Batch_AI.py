import logging
import os
import shutil
import random

from matplotlib import pyplot as plt
from matplotlib.pyplot import imshow

import numpy as np
import pandas as pd
from sklearn import datasets

import azureml.core
from azureml.core.experiment import Experiment
from azureml.core.workspace import Workspace
from azureml.train.automl import AutoMLConfig
from azureml.train.automl.run import AutoMLRun

print("Azure ML SDK Version: ", azureml.core.VERSION)

# Initialize Workspace
ws = Workspace.from_config()
print("Resource group: ", ws.resource_group)
print("Location: ", ws.location)
print("Workspace name: ", ws.name)

# Choose a name for the run history container in the workspace.
experiment_name = 'automl-remote-batchai'
project_folder = '../projects/automl-remote-batchai'

experiment = Experiment(ws, experiment_name)

# Create Batch AI Cluster
from azureml.core.compute import BatchAiCompute
from azureml.core.compute_target import ComputeTargetException

compute_target_name = 'myazbai'

try:
    batch_ai_compute = BatchAiCompute(workspace=ws, name=compute_target_name)
    print('found existing Azure Batch AI cluster:', batch_ai_compute.name)
except ComputeTargetException:
    print('creating new Azure Batch AI cluster...')
    batch_ai_config = BatchAiCompute.provisioning_configuration(
        vm_size="Standard_NC6",
        vm_priority="dedicated",
        autoscale_enabled = True,
        cluster_min_nodes = 0,
        cluster_max_nodes = 4
    )
    batch_ai_compute = BatchAiCompute.create(
        ws, 
        name=compute_target_name, 
        provisioning_configuration=batch_ai_config
    )
    batch_ai_compute.wait_for_completion(show_output=True)

project_folder = '../projects/automl-remote-batchai'
if not os.path.exists(project_folder):
    os.makedirs(project_folder)

shutil.copy('./scripts/get_data.py', project_folder)

print("Training the model...")
# configure Auto ML
automl_config = AutoMLConfig(
    task = 'classification',
    debug_log = 'automl_errors.log',
    primary_metric = 'AUC_weighted',
    max_time_sec = 120,
    iterations = 20,
    n_cross_validations = 5,
    preprocess = False,
    concurrent_iterations = 5,
    verbosity = logging.INFO,
    path = project_folder,
    compute_target = batch_ai_compute,
    data_script = project_folder + "/get_data.py"
)
remote_run = experiment.submit(automl_config, show_output = False)
remote_run.wait_for_completion(show_output = True)

# Retrieve All Child Runs
print("Retrieving All Child Runs")
children = list(remote_run.get_children())
metricslist = {}
for run in children:
    properties = run.get_properties()
    metrics = {k: v for k, v in run.get_metrics().items() if isinstance(v, float)}
    metricslist[int(properties['iteration'])] = metrics

rundata = pd.DataFrame(metricslist).sort_index(1)
print(rundata)

# Retrieving the Best Model
print("Retrieving the Best Model")
best_run, fitted_model = remote_run.get_output()
print(best_run)
print(fitted_model)

# Best Model Based on Any Other Metric
print("Retrieving the Best Model based on log_loss")
lookup_metric = "log_loss"
best_run, fitted_model = remote_run.get_output(metric = lookup_metric)
print(best_run)
print(fitted_model)

# Model from a Specific Iteration
print("Retrieving a Model from a Specific Iteration")
iteration = 3
third_run, third_model = remote_run.get_output(iteration=iteration)
print(third_run)
print(third_model)

# Testing the Fitted Model
print("Testing the Fitted Model")
digits = datasets.load_digits()
X_test = digits.data[:10, :]
y_test = digits.target[:10]
images = digits.images[:10]

# Randomly select digits and test.
for index in np.random.choice(len(y_test), 2, replace = False):
    print(index)
    predicted = fitted_model.predict(X_test[index:index + 1])[0]
    label = y_test[index]
    title = "Label value = %d  Predicted value = %d " % (label, predicted)
    fig = plt.figure(1, figsize=(3,3))
    ax1 = fig.add_axes((0,0,.8,.8))
    ax1.set_title(title)
    plt.imshow(images[index], cmap = plt.cm.gray_r, interpolation = 'nearest')
    plt.show()
