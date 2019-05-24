import os
import shutil
import time
import urllib.request
import requests
import json
import tensorflow as tf
import numpy as np
import azureml.core
from azureml.core import Workspace, Experiment
from azureml.core.compute import AksCompute, AmlCompute, ComputeTarget
from azureml.core.compute_target import ComputeTargetException
from azureml.core.runconfig import DataReferenceConfiguration
from azureml.core.model import Model
from azureml.core.conda_dependencies import CondaDependencies 
from azureml.train.dnn import TensorFlow
from azureml.core.webservice import AksWebservice, Webservice
from azureml.core.webservice.webservice import WebserviceException
from azureml.core.image import ContainerImage
from azureml.tensorboard import Tensorboard

# Check  SDK version numbers
print("Azure ML SDK Version: ", azureml.core.VERSION)
print("TensorFlow Version: ", tf.__version__)

# Initialize Workspace
ws = Workspace.from_config()
print("Resource group: ", ws.resource_group)
print("Location: ", ws.location)
print("Workspace name: ", ws.name)

experiment_name = 'image-retraining'
experiment = Experiment(workspace = ws, name = experiment_name)
global run

# Provision the AKS Cluster
print("Provisioning an AKS cluster...")

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

print("Provisioning an Azure ML Compute cluster ...")
# Create Azure ML Compute cluster (GPU-enabled) as a compute target
compute_target_name = 'myamlcompute'

try:
    aml_compute = AmlCompute(workspace=ws, name=compute_target_name)
    print('found existing Azure ML Compute cluster:', aml_compute.name)
except ComputeTargetException:
    print('creating new Azure ML Compute cluster...')
    aml_config = AmlCompute.provisioning_configuration(
        vm_size="Standard_NC6",
        vm_priority="dedicated",
        min_nodes = 0,
        max_nodes = 4,
        idle_seconds_before_scaledown=300
    )
    aml_compute = AmlCompute.create(
        ws, 
        name=compute_target_name, 
        provisioning_configuration=aml_config
    )
    aml_compute.wait_for_completion(show_output=True)

# Create a directory that will contain all the necessary code from your local machine 
# that you will need access to on the remote resource. This includes the training script, 
# and any additional files your training script depends on.
project_folder = './tmp/image_retraining'
os.makedirs(project_folder, exist_ok=True)
shutil.copy('./scripts/retrain.py', project_folder)

outputs_path = './outputs/image_retraining'
os.makedirs(outputs_path, exist_ok=True)
model_file = os.path.join(outputs_path, "output_graph.pb")
label_file = os.path.join(outputs_path, "output_labels.txt")

if os.path.exists(model_file) and os.path.exists(label_file):
    print("Local model found, skipping model training...")
else:
    # Upload data files into datastore
    # get the default datastore
    ds = ws.get_default_datastore()
    print("Datastore name: ", ds.name)
    print("Datastore type: ", ds.datastore_type)
    print("Account name: ", ds.account_name)
    print("Container name: ", ds.container_name)

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

    # Create a TensorFlow estimator
    # The AML SDK's TensorFlow estimator enables you to easily submit TensorFlow training
    # jobs for both single-node and distributed runs. 
    # For more information on the TensorFlow estimator, refer 
    # https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-train-tensorflow

    script_params={
        '--image_dir': str(ds.as_download()),
        '--summaries_dir': './logs',
        '--output_graph': './outputs/output_graph.pb',
        '--output_labels': './outputs/output_labels.txt',
        '--saved_model_dir': './outputs/model'
    }

    estimator = TensorFlow(
        source_directory=project_folder,
        source_directory_data_store=ds,
        compute_target=aml_compute,
        script_params=script_params,
        entry_script='retrain.py',
        pip_packages=['tensorflow_hub'],
        node_count=1,
        framework_version='1.10',
        use_gpu=True
    )

    # Overwrite data store reference
    dr = DataReferenceConfiguration(
        datastore_name=ds.name, 
        path_on_datastore='flower_photos', 
        mode='download', # download files from datastore to compute target
        overwrite=True
    )
    estimator.run_config.data_references[ds.name] = dr

    # Submit Experiment
    print("Training the model...")
    run = experiment.submit(estimator)
    run.wait_for_completion(show_output=True)

    print("Waiting for the run to complete...")
    status = run.get_status()
    while status != 'Completed' and status != 'Failed':
        print('current status: {} - waiting...'.format(run.get_status()))
        time.sleep(30)
        status = run.get_status()

    # Download results
    print("Downloading the results...")
    for filename in run.get_file_names():
        if filename.startswith('outputs'):
            print("downloading", filename, '...')
            run.download_file(
                filename, 
                output_file_path=outputs_path + filename.replace('outputs/','/')
            )

# Test model locally
print("Testing the model locally...")

def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())

    with graph.as_default():
        tf.import_graph_def(graph_def)

    return graph

def read_tensor_from_image_file(file_name,
                                input_height=299,
                                input_width=299,
                                input_mean=0,
                                input_std=255):
    input_name = "file_reader"
    file_reader = tf.read_file(file_name, input_name)
    if file_name.endswith(".png"):
        image_reader = tf.image.decode_png(file_reader, channels=3, name="png_reader")
    elif file_name.endswith(".gif"):
        image_reader = tf.squeeze(tf.image.decode_gif(file_reader, name="gif_reader"))
    elif file_name.endswith(".bmp"):
        image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
    else:
        image_reader = tf.image.decode_jpeg(file_reader, channels=3, name="jpeg_reader")
    float_caster = tf.cast(image_reader, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0)
    resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    with tf.Session() as sess:
        result = sess.run(normalized)
        return result

def load_labels(label_file):
    label = []
    proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
        label.append(l.rstrip())
    return label

graph = load_graph(model_file)

input_height = 299
input_width = 299
input_mean = 0
input_std = 255

input_layer = "Placeholder"
output_layer = "final_result"

def predict_flower(data):
    input_name = "import/" + input_layer
    output_name = "import/" + output_layer
    input_operation = graph.get_operation_by_name(input_name)
    output_operation = graph.get_operation_by_name(output_name)

    with tf.Session(graph=graph) as sess:
        results = sess.run(output_operation.outputs[0], {
            input_operation.outputs[0]: data
        })
    results = np.squeeze(results)

    top_k = results.argsort()[-5:][::-1]
    labels = load_labels(label_file)
    for i in top_k:
        print(labels[i], results[i])

print("Opening Daisy1.jpg...")
file_name = "./resources/test-images/Daisy1.jpg"

t = read_tensor_from_image_file(
    file_name,
    input_height=input_height,
    input_width=input_width,
    input_mean=input_mean,
    input_std=input_std
)

print("Predicting...")
predict_flower(t)

print("Opening Rose1.jpg...")
file_name = "./resources/test-images/Rose1.jpg"

t = read_tensor_from_image_file(
    file_name,
    input_height=input_height,
    input_width=input_width,
    input_mean=input_mean,
    input_std=input_std
)

print("Predicting...")
predict_flower(t)

service_name = 'flower-photos-svc'
try:
    service = AksWebservice(ws, service_name)
except WebserviceException:
    print("Registering the model...")
    model_graph_name = "flower_photos_graph"
    model_labels_name = "flower_photos_labels"

    model_graph = Model.register(
        model_path=model_file,
        model_name=model_graph_name,
        tags={"data": "flower_photos", "model": "classification"},
        description="Retrained Inception V3 model with flower photos",
        workspace=ws
    )

    model_labels = Model.register(
        model_path=label_file,
        model_name=model_labels_name,
        tags={"data": "flower_photos", "model": "classification"},
        description="Output labels of the retrained Inception V3 model with flower photos",
        workspace=ws
    )

    print("Checking AKS state...")
    status = aks_target.get_status()
    while status != 'Succeeded' and status != 'Failed':
        print('current status: {} - waiting...'.format(status))
        time.sleep(30)
        status = aks_target.get_status()

    print("Creating image and service configuration...")
    myenv = CondaDependencies()
    myenv.add_tensorflow_conda_package(core_type='cpu')
    myenv.add_conda_package("numpy")

    with open(os.path.join(project_folder, "myenv.yml"),"w") as f:
        f.write(myenv.serialize_to_string())

    shutil.copy("./scripts/score_flowers.py", './')

    # configure the image
    image_config = ContainerImage.image_configuration(
        execution_script="score_flowers.py", 
        runtime="python", 
        conda_file=os.path.join(project_folder, "myenv.yml")
    )

    aks_config = AksWebservice.deploy_configuration(
        cpu_cores=1, 
        memory_gb=1, 
        #collect_model_data=True,
        enable_app_insights=True, 
        tags={"data": "flower_photos",  "method" : "TensorFlow"}, 
        description='Predict flowers with TensorFlow'
    )

    print("Creating the image and deploy as web service...")
    service = Webservice.deploy_from_model(
        workspace=ws,
        name=service_name,
        deployment_config=aks_config,
        deployment_target=aks_target,
        models=[model_graph, model_labels],
        image_config=image_config
    )
    service.wait_for_deployment(show_output=True)
    print(service.state)

print("Service URI:", service.scoring_uri)

print("Testings web service via SDK...")
file_name = "./resources/test-images/Daisy1.jpg"

for dirpath, dnames, fnames in os.walk("./resources/test-images/"):
    for f in fnames:
        file_name = os.path.join(dirpath, f)
        
        # load image
        print("Loading image", file_name)

        data = read_tensor_from_image_file(
            file_name,
            input_height=input_height,
            input_width=input_width,
            input_mean=input_mean,
            input_std=input_std
        )
        raw_data = str(data.tolist())

        # predict using the deployed model
        print("Sending image", f, "to service")
        response = service.run(input_data=raw_data)
        print("Service response:", response)
        print()

print("Testings web service via HTTP call...")
api_keys = service.get_keys()
headers = {
    'Content-Type':'application/json',
    'Authorization':('Bearer '+ api_keys[0])
}

file_name = "./resources/test-images/Daisy1.jpg"
data = read_tensor_from_image_file(
    file_name,
    input_height=input_height,
    input_width=input_width,
    input_mean=input_mean,
    input_std=input_std
)
input_data = str(data.tolist())

print("POST to url", service.scoring_uri)
resp = requests.post(service.scoring_uri, input_data, headers=headers)
print("prediction:", resp.text)

print("Deleting deployed flowers services...")
# Cleanup
service.delete()
if os.path.exists('score_flowers.py'):
    os.remove('score_flowers.py')

if not 'run' in globals():
    for r in experiment.get_runs():
        run_details = r.get_details()
        if run_details['status'] == 'Completed':
            run = r
            break

print('Starting TensorBoard...')
# The Tensorboard constructor takes an array of runs, so be sure and pass it in as a single-element array here
tb = Tensorboard([run])

# If successful, start() returns a string with the URI of the instance.
tb.start()
input("Press Enter to close TensorBoard service...")
tb.stop()
