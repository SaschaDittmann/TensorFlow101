# Check core SDK version number
import azureml.core
print("SDK version:", azureml.core.VERSION)

# Initialize Workspace
from azureml.core import Workspace

ws = Workspace.from_config()
print("Resource group: ", ws.resource_group)
print("Location: ", ws.location)
print("Workspace name: ", ws.name)

# Create a directory that will contain all the necessary code from your local machine 
# that you will need access to on the remote resource. This includes the training script, 
# and any additional files your training script depends on.
import os

project_folder = './tmp/fashion-mnist-aci'
os.makedirs(project_folder, exist_ok=True)

config_folder = os.path.join(project_folder, 'aml_config')
os.makedirs(config_folder, exist_ok=True)

import shutil
shutil.copy('./scripts/train_Fashion_MNIST.py', project_folder)
shutil.copy('./resources/configs/fashion_mnist_aci_dependencies.yml', config_folder)
shutil.copy('./resources/configs/fashion_mnist_aci.runconfig', config_folder)

# Create An Experiment
from azureml.core import Experiment
experiment_name = 'fashion-mnist'
experiment = Experiment(workspace = ws, name = experiment_name)

# Load Configuration
from azureml.core.runconfig import RunConfiguration
from azureml.core.conda_dependencies import CondaDependencies

# create a new runconfig object
run_config = RunConfiguration.load(path=project_folder, name='fashion_mnist_aci')

# Submit Experiment
from azureml.core.script_run_config import ScriptRunConfig

script_run_config = ScriptRunConfig(source_directory=project_folder,
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
run.download_file('outputs/saved_model.tar.gz', output_file_path='./outputs')
shutil.unpack_archive('./outputs/saved_model.tar.gz', './models/fashion_mnist/' + run.id.split('_')[1])
os.remove(path='./outputs/saved_model.tar.gz')