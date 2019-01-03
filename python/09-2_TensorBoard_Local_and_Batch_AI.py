# Check core SDK version number
import azureml.core
print("SDK version:", azureml.core.VERSION)

# Opt-in diagnostics for better experience, quality, and security of future releases.
from azureml.telemetry import set_diagnostics_collection
set_diagnostics_collection(send_diagnostics = True)

# Initialize Workspace
from azureml.core import Workspace

ws = Workspace.from_config()
print("Resource group: ", ws.resource_group)
print("Location: ", ws.location)
print("Workspace name: ", ws.name)

from os import path, makedirs
experiment_name = 'tensorboard-demo'

# experiment folder
import os
exp_dir = os.path.abspath(os.path.join('./outputs/', experiment_name))

if not path.exists(exp_dir):
    makedirs(exp_dir)

# runs we started in this session, for the finale
runs = []

# Create a run configuration.
from azureml.core.runconfig import RunConfiguration
run_config = RunConfiguration()
run_config.environment.python.user_managed_dependencies = True

# You can choose a specific Python environment by pointing to a Python path 
#run_config.environment.python.interpreter_path = '/home/ninghai/miniconda3/envs/sdk2/bin/python'

from azureml.core import Experiment, Run
from azureml.core.script_run_config import ScriptRunConfig
import tensorflow as tf

logs_dir = os.path.join(os.curdir, "logs")
data_dir = os.path.abspath(os.path.join(os.curdir, "mnist_data"))

if not path.exists(data_dir):
    makedirs(data_dir)

os.environ["TEST_TMPDIR"] = data_dir

# Writing logs to ./logs results in their being uploaded to Artifact Service,
# and thus, made accessible to our TensorBoard instance.
arguments_list = ["--log_dir", logs_dir, "--data_dir", data_dir]

# Create an experiment
exp = Experiment(ws, experiment_name)

# If you would like the run to go for longer, add --max_steps 5000 to the arguments list:
# arguments_list += ["--max_steps", "5000"]

script = ScriptRunConfig('./scripts',
                         script="mnist_with_summaries.py",
                         run_config=run_config,
                         arguments=arguments_list)

run = exp.submit(script)
runs.append(run)

run.wait_for_completion(show_output=True)

# Once more, with a Batch AI cluster
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
        min_nodes = 0,
        max_nodes = 4,
        idle_seconds_before_scaledown=300
    )
    batch_ai_compute = AmlCompute.create(
        ws, 
        name=compute_target_name, 
        provisioning_configuration=batch_ai_config
    )
    batch_ai_compute.wait_for_completion(show_output=True)

# Submit run using TensorFlow estimator
from azureml.train.dnn import TensorFlow

script_params = {
    "--log_dir": "./logs"
}

tf_estimator = TensorFlow(
    source_directory='./scripts',
    compute_target=batch_ai_compute,
    entry_script='mnist_with_summaries.py',
    script_params=script_params
)

run = exp.submit(tf_estimator)

runs.append(run)
run.wait_for_completion(show_output=True)

from azureml.contrib.tensorboard import Tensorboard

# The TensorBoard constructor takes an array of runs...
# and it turns out that we have been building one of those all along.
tb = Tensorboard(runs)

# If successful, start() returns a string with the URI of the instance.
tb.start()
input("Press Enter to close TensorBoard service...")
tb.stop()
