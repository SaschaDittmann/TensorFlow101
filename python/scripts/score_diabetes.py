import pickle
import json
import numpy 
from sklearn.externals import joblib
from sklearn.linear_model import Ridge
from azureml.core.model import Model

from azureml.monitoring import ModelDataCollector
import time

def init():
    global model
    print ("model initialized" + time.strftime("%H:%M:%S"))
    # note here "sklearn_regression_model.pkl" is the name of the model registered under the workspace
    # this call should return the path to the model.pkl file on the local disk.
    model_path = Model.get_model_path(model_name = 'sklearn_regression_model')
    # deserialize the model file back into a sklearn model
    model = joblib.load(model_path)
    
    global inputs_dc, prediction_dc
    # this setup will help us save our inputs under the "inputs" path in our Azure Blob
    inputs_dc = ModelDataCollector(model_name="sklearn_regression_model", identifier="inputs", feature_names=["feat1", "feat2"]) 
    # this setup will help us save our ipredictions under the "predictions" path in our Azure Blob
    prediction_dc = ModelDataCollector("sklearn_regression_model", identifier="predictions", feature_names=["prediction1", "prediction2"])

def run(raw_data):
    global inputs_dc, prediction_dc
    try:
        data = json.loads(raw_data)['data']
        data = numpy.array(data)
        print ("saving input data" + time.strftime("%H:%M:%S"))
        inputs_dc.collect(data) #this call is saving our input data into our blob
        
        result = model.predict(data)
        print ("saving prediction data" + time.strftime("%H:%M:%S"))
        prediction_dc.collect(result)#this call is saving our prediction data into our blob
        
        # you can return any data type as long as it is JSON-serializable
        return result.tolist()
    except Exception as e:
        error = str(e)
        print (error + time.strftime("%H:%M:%S"))
        return error
