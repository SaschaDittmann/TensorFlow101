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

from azureml.core.runconfig import RunConfiguration
from azureml.core.conda_dependencies import CondaDependencies

# create a new runconfig object
run_config = RunConfiguration()

# signal that you want to use ACI to execute script.
run_config.target = "containerinstance"

# ACI container group is only supported in certain regions, which can be different than the region the Workspace is in.
run_config.container_instance.region = ws.location

# set the ACI CPU and Memory 
run_config.container_instance.cpu_cores = 1
run_config.container_instance.memory_gb = 2

# enable Docker 
run_config.environment.docker.enabled = True

# set Docker base image to the default CPU-based image
run_config.environment.docker.base_image = azureml.core.runconfig.DEFAULT_CPU_IMAGE

# use conda_dependencies.yml to create a conda environment in the Docker image for execution
run_config.environment.python.user_managed_dependencies = False

# auto-prepare the Docker image when used for execution (if it is not already prepared)
run_config.auto_prepare_environment = True

# specify CondaDependencies obj
conda_dep = CondaDependencies.create(
    python_version='3.6.2', 
    conda_packages=['keras', 'matplotlib']
)
conda_dep.add_tensorflow_conda_package(core_type='cpu')
run_config.environment.python.conda_dependencies = conda_dep

from azureml.core.script_run_config import ScriptRunConfig

script_run_config = ScriptRunConfig(source_directory='./scripts',
                                    script='train_Fashion_MNIST.py',
                                    run_config=run_config)

run = experiment.submit(script_run_config)
run.tag("Description","ACI trained Fashion MNIST model")
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
