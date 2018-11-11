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
experiment_name = 'image-retraining'
experiment = Experiment(workspace = ws, name = experiment_name)

# Create Azure Batch AI cluster (GPU-enabled) as a compute target
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

# Upload data files into datastore
# get the default datastore
ds = ws.get_default_datastore()
print("Datastore name: ", ds.name)
print("Datastore type: ", ds.datastore_type)
print("Account name: ", ds.account_name)
print("Container name: ", ds.container_name)

import os
import shutil
import urllib.request

tmp_path = './tmp/datastore'
os.makedirs(tmp_path, exist_ok=True)

print('Downloading flower photos...')
urllib.request.urlretrieve("http://download.tensorflow.org/example_images/flower_photos.tgz", tmp_path + "/flower_photos.tgz")
print('Unpacking archive...')
shutil.unpack_archive(tmp_path + '/flower_photos.tgz', tmp_path)

images_path = tmp_path + '/flower_photos/'
for (dirpath, dirnames, filenames) in os.walk(images_path):
    print('Uploading', dirpath, '...')
    ds.upload_files(
        [dirpath + '/' + f for f in filenames], 
        target_path=dirpath.replace(tmp_path + '/', ''), 
        overwrite=True
    )

# Create a directory that will contain all the necessary code from your local machine 
# that you will need access to on the remote resource. This includes the training script, 
# and any additional files your training script depends on.
project_folder = './tmp/image_retraining'
os.makedirs(project_folder, exist_ok=True)
shutil.copy('./scripts/retrain.py', project_folder)

# Create a TensorFlow estimator
# The AML SDK's TensorFlow estimator enables you to easily submit TensorFlow training
# jobs for both single-node and distributed runs. 
# For more information on the TensorFlow estimator, refer 
# https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-train-tensorflow
from azureml.train.dnn import TensorFlow
from azureml.core.runconfig import DataReferenceConfiguration

script_params={
    '--image_dir': str(ds.as_download()),
    '--summaries_dir': './outputs/retrain_logs',
    '--output_graph': './outputs/output_graph.pb',
    '--output_labels': './outputs/output_labels.txt',
    '--saved_model_dir': './outputs/model'
}

estimator = TensorFlow(
  source_directory=project_folder,
  source_directory_data_store=ds,
  compute_target=batch_ai_compute,
  script_params=script_params,
  entry_script='retrain.py',
  pip_packages=['tensorflow_hub'],
  node_count=1,
  use_gpu=True
)

# Overwrite data store reference
dr = DataReferenceConfiguration(
    datastore_name=ds.name, 
    path_on_datastore='flower_photos', 
    mode='download', # download files from datastore to compute target
    overwrite=True
)
estimator.run_config.data_references['workspacefilestore'] = dr

# Submit Experiment
run = experiment.submit(estimator)
run.wait_for_completion(show_output=True)

# Download results
import time

status = run.get_status()
while status != 'Completed' and status != 'Failed':
    print('current status: {} - waiting...'.format(run.get_status()))
    time.sleep(10)
    status = run.get_status()

outputs_path = './outputs/image_retraining'

os.makedirs(outputs_path, exist_ok=True)

for filename in run.get_file_names():
    if filename.startswith('outputs'):
        print("downloading", filename, '...')
        run.download_file(
            filename, 
            output_file_path=outputs_path + filename.replace('outputs/','/')
        )
print('completed')
