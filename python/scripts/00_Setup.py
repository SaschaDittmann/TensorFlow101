from azureml.core import Workspace
import os
import sys

subscription_id = os.environ.get("SUBSCRIPTION_ID", "<subscription_id>")
resource_group = os.environ.get("RESOURCE_GROUP", "tensorflow101")
workspace_name = os.environ.get("WORKSPACE_NAME", "tensorflow101-mlwrksp")
workspace_region = os.environ.get("WORKSPACE_REGION", "westeurope")

try:
    ws = Workspace(subscription_id = subscription_id, resource_group = resource_group, workspace_name = workspace_name)
    ws.write_config()
    print('Library configuration succeeded')
except:
    ws = Workspace.create(name = workspace_name,
                      subscription_id = subscription_id,
                      resource_group = resource_group, 
                      location = workspace_region,
                      create_resource_group = True,
                      exist_ok = True)
    ws.get_details()
    ws.write_config()
    print('Library configuration succeeded')