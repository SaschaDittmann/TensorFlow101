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
from azureml.core.compute import AmlCompute
from azureml.core.compute_target import ComputeTargetException

compute_target_name = 'myazbai'

try:
    batch_ai_compute = AmlCompute(workspace=ws, name=compute_target_name)
    print('found existing Azure Batch AI cluster:', batch_ai_compute.name)
except ComputeTargetException:
    print('creating new Azure Batch AI cluster...')
    batch_ai_config = AmlCompute.provisioning_configuration(
        vm_size="Standard_NC6",
        vm_priority="dedicated",
        autoscale_enabled = True,
        min_nodes = 0,
        max_nodes = 4
    )
    batch_ai_compute = AmlCompute.create(
        ws, 
        name=compute_target_name, 
        provisioning_configuration=batch_ai_config
    )
    batch_ai_compute.wait_for_completion(show_output=True)

# Create a directory that will contain all the necessary code from your local machine 
# that you will need access to on the remote resource. This includes the training script, 
# and any additional files your training script depends on.
import os

project_folder = './tmp/tf-distr-ps'
os.makedirs(project_folder, exist_ok=True)

import shutil
shutil.copy('./scripts/tf_mnist_replica.py', project_folder)

from azureml.train.dnn import TensorFlow

script_params={
    '--num_gpus': 1
}

estimator = TensorFlow(source_directory=project_folder,
                       compute_target=batch_ai_compute,
                       script_params=script_params,
                       entry_script='tf_mnist_replica.py',
                       node_count=2,
                       worker_count=2,
                       parameter_server_count=1,   
                       distributed_backend='ps',
                       use_gpu=True)

run = experiment.submit(estimator)
print(run.get_details())

run.wait_for_completion(show_output=True)
