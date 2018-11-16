import json
import os
import traceback
import numpy as np
import tensorflow as tf
import time

from azureml.core.model import Model

def load_graph(graph_path):
    global graph
    global input_operation
    global output_operation

    print("loading graph from", graph_path, time.strftime("%H:%M:%S"))
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(graph_path, "rb") as f:
        graph_def.ParseFromString(f.read())

    with graph.as_default():
        tf.import_graph_def(graph_def)

    input_operation = graph.get_operation_by_name('import/Placeholder')
    output_operation = graph.get_operation_by_name('import/final_result')
    print("graph loaded successfully.", time.strftime("%H:%M:%S"))

def load_labels(label_path):
    global labels
    
    print("loading labels from", label_path, time.strftime("%H:%M:%S"))
    labels = []
    proto_as_ascii_lines = tf.gfile.GFile(label_path).readlines()
    for l in proto_as_ascii_lines:
        labels.append(l.rstrip())
    print("labels loaded successfully.", time.strftime("%H:%M:%S"))

def init():
    try:
        print ("model initializing" + time.strftime("%H:%M:%S"))
        # retreive the path to the model file using the model name
        graph_path = Model.get_model_path('flower_photos_graph')
        load_graph(graph_path)

        labels_path = Model.get_model_path('flower_photos_labels')
        load_labels(labels_path)
        print ("model initialized" + time.strftime("%H:%M:%S"))
    except Exception as e:
        error = str(e)
        stacktrace = traceback.format_exc()
        print (error + time.strftime("%H:%M:%S"))
        print (stacktrace)
        raise

def run(raw_data):
    try:
        data = json.loads(raw_data)
        data = np.array(data)
        print ("image array: " + str(data)[:50])
        
        # make prediction
        with tf.Session(graph=graph) as sess:
            results = sess.run(output_operation.outputs[0], {
                input_operation.outputs[0]: data
            })
        results = np.squeeze(results)

        top_k = results.argsort()[-5:][::-1]

        result = []
        for i in top_k:
            result.append([labels[i], results[i]])
        print ("result: " + str(result))
        
        # you can return any data type as long as it is JSON-serializable
        return str(result)
    except Exception as e:
        error = str(e)
        stacktrace = traceback.format_exc()
        print (error + time.strftime("%H:%M:%S"))
        print (stacktrace)
        return stacktrace
