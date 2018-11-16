import azureml
from azureml.core import Workspace

# display the core SDK version number
print("Azure ML SDK Version: ", azureml.core.VERSION)

ws = Workspace.from_config()
print("Resource group: ", ws.resource_group)
print("Location: ", ws.location)
print("Workspace name: ", ws.name)

from azureml.core.webservice import Webservice

for web_svc in Webservice.list(ws):
    print("Deleting web service", web_svc.name, "...")
    web_svc.delete()

from azureml.core import ComputeTarget

for target in ComputeTarget.list(ws):
    print("Deleting compute target", target.name, "...")
    target.delete()

from azureml.core import Image

for img in Image.list(ws):
    print("Deleting image", img.id, "...")
    img.delete()

from azureml.core.model import Model

for model in Model.list(ws):
    print("Deleting model", model.id, "...")
    model.delete()
