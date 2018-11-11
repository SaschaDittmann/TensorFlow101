# Check core SDK version number
import azureml.core
print("SDK version:", azureml.core.VERSION)

# Initialize Workspace
from azureml.core import Workspace

ws = Workspace.from_config()
print("Resource group: ", ws.resource_group)
print("Location: ", ws.location)
print("Workspace name: ", ws.name)

from azureml.core import Experiment
experiment_name = 'fashion-mnist'
experiment = Experiment(workspace = ws, name = experiment_name)

# Create Azure Batch AI cluster (GPU-enabled) as a compute target
from azureml.core.compute import BatchAiCompute
from azureml.core.compute_target import ComputeTargetException

compute_target_name = 'myazbai'

try:
    batch_ai_compute = BatchAiCompute(workspace=ws, name=compute_target_name)
    print('found existing:', batch_ai_compute.name)
except ComputeTargetException:
    print('creating new.')
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

# Create a TensorFlow estimator
# The AML SDK's TensorFlow estimator enables you to easily submit TensorFlow training
# jobs for both single-node and distributed runs. 
# For more information on the TensorFlow estimator, refer 
# https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-train-tensorflow
from azureml.train.dnn import TensorFlow

estimator = TensorFlow(source_directory='./scripts',
                       compute_target=batch_ai_compute,
                       entry_script='train_Fashion_MNIST.py',
                       node_count=1,
                       worker_count=1,
                       parameter_server_count=1,
                       conda_packages=['keras', 'matplotlib'],
                       use_gpu=True)

# Submit Experiment
run = experiment.submit(estimator)
run.tag("Description","Batch AI trained Fashion MNIST model")
run.wait_for_completion(show_output=True)

# Show Metrics
# get all metris logged in the run
run.get_metrics()
metrics = run.get_metrics()

import numpy as np
print('loss is {0:.2f}, and accuracy is {1:0.2f}'.format(
    metrics['loss'], 
    metrics['accuracy']
))

# Plot data to see relationships in training and validation data
import numpy as np
import matplotlib.pyplot as plt
epoch_list = list(range(1, len(metrics['Training Accuracy']) + 1))  # values for x axis [1, 2, ..., # of epochs]
plt.plot(epoch_list, metrics['Training Accuracy'], epoch_list, metrics['Validation Accuracy'])
plt.legend(('Training Accuracy', 'Validation Accuracy'))
plt.show()

# show all the files stored within the run record
print('files stored within the run record:')
run.get_file_names()

# Download Model
import os
import shutil

os.makedirs('./outputs', exist_ok=True)
os.makedirs('./models', exist_ok=True)

print('downloading model...')
run.download_file('outputs/saved_model.zip', output_file_path='./outputs')
shutil.unpack_archive('./outputs/saved_model.zip', './models/fashion_mnist/' + run.id.split('_')[1])
os.remove(path='./outputs/saved_model.zip')
