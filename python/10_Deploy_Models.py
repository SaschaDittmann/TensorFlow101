import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
 
import azureml
from azureml.core import Workspace, Run

# display the core SDK version number
print("TensorFlow Version: ", tf.__version__)
print("Azure ML SDK Version: ", azureml.core.VERSION)

# Initialize Workspace
ws = Workspace.from_config()
print("Resource group: ", ws.resource_group)
print("Location: ", ws.location)
print("Workspace name: ", ws.name)

import os

project_folder = './tmp/deploy_sklearn_mnist'
os.makedirs(project_folder, exist_ok=True)

# Register a model
print("Registering MNIST model...")
from azureml.core.model import Model

model_name = "sklearn_mnist"
model_path = "./resources/models/sklearn_mnist_model.pkl"

model = Model.register(
    model_path=model_path,
    model_name=model_name,
    tags={"data": "mnist", "model": "classification"},
    description="Mnist handwriting recognition",
    workspace=ws
)

# Download MNIST Dataset
print("Downloading MNIST data...")
from tensorflow.examples.tutorials.mnist import input_data

# Supress warning and informational messages
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)

mnist = input_data.read_data_sets("./MNIST_data/")

# Restore warning and informational messages
tf.logging.set_verbosity(old_v)

print('images shape:', mnist.train.images.shape)
print('labels shape:', mnist.train.labels.shape)

# Test model locally
print("Testing downloaded model...")
import pickle
from sklearn.externals import joblib

clf = joblib.load(model_path)
y_hat = clf.predict(mnist.test.images)

#  Examine the confusion matrix
from sklearn.metrics import confusion_matrix

conf_mx = confusion_matrix(mnist.test.labels, y_hat)
print(conf_mx)
print('Overall accuracy:', np.average(y_hat == mnist.test.labels))

# normalize the diagnal cells so that they don't overpower the rest of the cells when visualized
row_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx / row_sums
np.fill_diagonal(norm_conf_mx, 0)

fig = plt.figure(figsize=(8,5))
ax = fig.add_subplot(111)
cax = ax.matshow(norm_conf_mx, cmap=plt.cm.bone)
ticks = np.arange(0, 10, 1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(ticks)
ax.set_yticklabels(ticks)
fig.colorbar(cax)
plt.ylabel('true labels', fontsize=14)
plt.xlabel('predicted values', fontsize=14)
#plt.savefig(os.path.join(project_folder, 'conf.png'))
plt.show()

# Provision the AKS Cluster
print("Provisioning an AKS cluster...")
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
        agent_count=3,
        vm_size="Standard_B4ms"
    )
    
    # Create the cluster
    aks_target = ComputeTarget.create(
        workspace = ws, 
        name = aks_name, 
        provisioning_configuration = prov_config
    )

print("Creating docker image configuration...")
from azureml.core.conda_dependencies import CondaDependencies 

myenv = CondaDependencies()
myenv.add_conda_package("scikit-learn")

with open(os.path.join(project_folder, "myenv.yml"),"w") as f:
    f.write(myenv.serialize_to_string())

import shutil
shutil.copy("./scripts/score_mnist.py", './')

# Create ACI configuration
from azureml.core.webservice import AciWebservice

aciconfig = AciWebservice.deploy_configuration(
    cpu_cores=1, 
    memory_gb=1, 
    tags={"data": "MNIST",  "method" : "sklearn"}, 
    description='Predict MNIST with sklearn'
)

# Create Docker Image
from azureml.core.image import ContainerImage

# configure the image
image_config = ContainerImage.image_configuration(
    execution_script="score_mnist.py", 
    runtime="python", 
    conda_file=os.path.join(project_folder, "myenv.yml")
)

image = ContainerImage.create(
    name = "sklearn-mnist-img",
    models = [model],
    image_config = image_config,
    workspace = ws
)

image.wait_for_creation(show_output = True)

# Deploy the image in ACI
print("Deploying the service as Azure Container Instance...")
from azureml.core.webservice import Webservice

aci_service = Webservice.deploy_from_image(
    workspace = ws, 
    name = 'sklearn-mnist-aci-svc',
    image = image,
    deployment_config = aciconfig
)

aci_service.wait_for_deployment(show_output=True)

print("ACI Score URL:", aci_service.scoring_uri)

print("Testing deployed service via SDK...")
# Test deployed service
import json

# find 30 random samples from test set
n = 30
sample_indices = np.random.permutation(mnist.test.images.shape[0])[0:n]

test_samples = json.dumps({"data": mnist.test.images[sample_indices].tolist()})
test_samples = bytes(test_samples, encoding='utf8')

# predict using the deployed model
result = aci_service.run(input_data=test_samples)

# compare actual value vs. the predicted values:
i = 0
plt.figure(figsize = (20, 1))

for s in sample_indices:
    plt.subplot(1, n, i + 1)
    plt.axhline('')
    plt.axvline('')
    
    # use different color for misclassified sample
    font_color = 'red' if mnist.test.labels[s] != result[i] else 'black'
    clr_map = plt.cm.gray if mnist.test.labels[s] != result[i] else plt.cm.Greys
    
    plt.text(x=10, y =-10, s=result[i], fontsize=18, color=font_color)
    plt.imshow(mnist.test.images[s].reshape(28, 28), cmap=clr_map)
    
    i = i + 1
plt.show()

print("Testing deployed service via HTTP call...")
# You can also send raw HTTP request to test the web service.
import requests
import json

# send a random row from the test set to score
random_index = np.random.randint(0, len(mnist.test.images)-1)
input_data = "{\"data\": [" + str(list(mnist.test.images[random_index])) + "]}"

headers = {'Content-Type':'application/json'}

resp = requests.post(aci_service.scoring_uri, input_data, headers=headers)

print("POST to url", aci_service.scoring_uri)
#print("input data:", input_data)
print("label:", mnist.test.labels[random_index])
print("prediction:", resp.text)

print("Deploying the service to Azure Kubernetes Service...")
# Deploy to Azure Kubernetes Services

# Check AKS cluster state
import time

status = aks_target.get_status()
while status != 'Succeeded' and status != 'Failed':
    print('current status: {} - waiting...'.format(status))
    time.sleep(10)
    status = aks_target.get_status()

from azureml.core.webservice import Webservice, AksWebservice

aks_service_name = 'sklearn-mnist-aks-svc'

# Set the web service configuration (using default here)
aks_config = AksWebservice.deploy_configuration(
    cpu_cores=1, 
    memory_gb=1, 
    tags={"data": "MNIST",  "method" : "sklearn"}, 
    description='Predict MNIST with sklearn'
)

aks_service = Webservice.deploy_from_image(
    workspace = ws, 
    name = aks_service_name,
    image = image,
    deployment_config = aks_config,
    deployment_target = aks_target
)

aks_service.wait_for_deployment(show_output = True)
print(aks_service.state)

print("Testing deployed service via HTTP call...")
import requests
import json

# send a random row from the test set to score
random_index = np.random.randint(0, len(mnist.test.images)-1)
input_data = "{\"data\": [" + str(list(mnist.test.images[random_index])) + "]}"

# for AKS deployment you'd need to the service key in the header as well
api_keys = aks_service.get_keys()
headers = {'Content-Type':'application/json',  'Authorization':('Bearer '+ api_keys[0])} 

resp = requests.post(aks_service.scoring_uri, input_data, headers=headers)

print("POST to url", aks_service.scoring_uri)
#print("input data:", input_data)
print("label:", mnist.test.labels[random_index])
print("prediction:", resp.text)

print("Deleting deployed MNIST services...")
# Cleanup
aci_service.delete()
aks_service.delete()
if os.path.exists('score_mnist.py'):
    os.remove('score_mnist.py')
