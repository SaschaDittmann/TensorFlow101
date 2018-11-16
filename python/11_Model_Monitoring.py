# Import your dependencies
from azureml.core import Workspace, Run
from azureml.core.compute import AksCompute, ComputeTarget
from azureml.core.webservice import Webservice, AksWebservice
from azureml.core.image import Image
from azureml.core.model import Model

import azureml.core
print("Azure ML SDK Version: ", azureml.core.VERSION)

# Initialize Workspace
ws = Workspace.from_config()
print("Resource group: ", ws.resource_group)
print("Location: ", ws.location)
print("Workspace name: ", ws.name)

import os

project_folder = './tmp/model_monitoring'
os.makedirs(project_folder, exist_ok=True)

print("Registering diabetes model...")
# Register Model
model = Model.register(
    model_path = "./resources/models/sklearn_regression_model.pkl", # this points to a local file
    model_name = "sklearn_regression_model", # this is the name the model is registered as
    tags = {'area': "diabetes", 'type': "regression"},
    description = "Ridge regression model to predict diabetes",
    workspace = ws
)

print("Creating docker image configuration...")
# Update your myenv.yml file with the required module
from azureml.core.conda_dependencies import CondaDependencies 

myenv = CondaDependencies.create(conda_packages=['numpy','scikit-learn'])
myenv.add_pip_package("azureml-monitoring")

with open(os.path.join(project_folder, "myenv.yml"),"w") as f:
    f.write(myenv.serialize_to_string())

import shutil
shutil.copy("./scripts/score_diabetes.py", './')

from azureml.core.image import ContainerImage

# Create your new Image
image_config = ContainerImage.image_configuration(
    execution_script = "score_diabetes.py",
    runtime = "python",
    conda_file = os.path.join(project_folder, "myenv.yml"),
    description = "Image with ridge regression model",
    tags = {'area': "diabetes", 'type': "regression"}
)

image = ContainerImage.create(
    name = "diabetes-model",
    models = [model],
    image_config = image_config,
    workspace = ws
)

image.wait_for_creation(show_output = True)

print("Provisioning an AKS cluster...")
# Provision the AKS Cluster
from azureml.core.compute import AksCompute, ComputeTarget
from azureml.core.compute_target import ComputeTargetException

aks_name = 'myaks'

try:
    aks_target = AksCompute(workspace=ws, name=aks_name)
    print('found existing Azure Kubernetes Service:', aks_target.name)
except ComputeTargetException:
    print('creating new Azure Kubernetes Service.')

    # AKS configuration
    prov_config = AksCompute.provisioning_configuration(
        #agent_count=2,
        #vm_size="Standard_D3_v2"
    )
    
    # Create the cluster
    aks_target = ComputeTarget.create(
        workspace = ws, 
        name = aks_name, 
        provisioning_configuration = prov_config
    )

aks_target.wait_for_completion(show_output = True)
print(aks_target.provisioning_state)
print(aks_target.provisioning_errors)

# Check AKS cluster state
import time

status = aks_target.get_status()
while status != 'Succeeded' and status != 'Failed':
    print('current status: {} - waiting...'.format(status))
    time.sleep(10)
    status = aks_target.get_status()

print("Deploying the service to Azure Kubernetes Service...")
# Activate Data Collection and App Insights through updating AKS Webservice configuration
aks_config = AksWebservice.deploy_configuration(
    collect_model_data=True, 
    enable_app_insights=True
)

# Deploy your service
aks_service_name ='diabetes-aks-svc'

aks_service = Webservice.deploy_from_image(
    workspace = ws, 
    name = aks_service_name,
    image = image,
    deployment_config = aks_config,
    deployment_target = aks_target
)

aks_service.wait_for_deployment(show_output = True)
print(aks_service.state)

print("Testing deployed service via SDK...")
# Test Service
import json

test_sample = json.dumps({'data': [
    [1,2,3,4,54,6,7,8,88,10], 
    [10,9,8,37,36,45,4,33,2,1]
]})
test_sample = bytes(test_sample,encoding = 'utf8')

prediction = aks_service.run(input_data = test_sample)
print("POST to url", aks_service.scoring_uri)
print("input data:", test_sample)
print("prediction:", prediction)
print()

print("Testing deployed service via HTTP call...")
import requests
import json

api_keys = aks_service.get_keys()
headers = {
    'Content-Type':'application/json',
    'Authorization':('Bearer '+ api_keys[0])
}

test_sample = json.dumps({'data': [
    [1,2,3,4,54,6,7,8,88,10], 
    [10,9,8,37,36,45,4,33,2,1]
]})

print("POST to url", aks_service.scoring_uri)
print("input data:", test_sample)
resp = requests.post(aks_service.scoring_uri, test_sample, headers=headers)
print("prediction:", resp.text)

# Cleanup
print("Deleting deployed diabetes service...")
aks_service.delete()
if os.path.exists('score_diabetes.py'):
    os.remove('score_diabetes.py')
