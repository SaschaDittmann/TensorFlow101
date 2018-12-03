# Workshop: TensorFlow 101 with the Azure Machine Learning Service

[![](https://img.shields.io/github/license/mashape/apistatus.svg)](https://github.com/SaschaDittmann/TensorFlow101)  [![](https://img.shields.io/github/tag/SaschaDittmann/TensorFlow101.svg)](https://github.com/SaschaDittmann/TensorFlow101)

## About this Docker Image

This Docker Image has been created for the "TensorFlow 101 with the Azure Machine Learning service" workshop.

It's based on the official [tensorflow docker image](https://hub.docker.com/r/tensorflow/tensorflow/) and adds the [Azure Machine Learning SDK for Python](https://docs.microsoft.com/en-us/python/api/overview/azure/ml/intro?view=azure-ml-py), as well as the jupyter notebooks required for the workshop.

[![](https://img.shields.io/docker/pulls/bytesmith/tensorflow101.svg)](https://hub.docker.com/r/bytesmith/tensorflow101) [![](https://img.shields.io/docker/automated/bytesmith/tensorflow101.svg)](https://hub.docker.com/r/bytesmith/tensorflow101/builds)

version          | description                               | size 
---------------- | ----------------------------------------- | ------
[![](https://images.microbadger.com/badges/version/bytesmith/tensorflow101.svg)](https://hub.docker.com/r/bytesmith/tensorflow101) | Latest build from the [GitHub Repo](https://github.com/SaschaDittmann/TensorFlow101) | [![](https://images.microbadger.com/badges/image/bytesmith/tensorflow101.svg)](https://microbadger.com/images/bytesmith/tensorflow101)
[![](https://images.microbadger.com/badges/version/bytesmith/tensorflow101:release-0.2.0.svg)](https://hub.docker.com/r/bytesmith/tensorflow101) | Demos and exercises, used at the at the MS Partner Training (Nov 23th, 2018). - [GitHub Release](https://github.com/SaschaDittmann/TensorFlow101/releases/tag/0.2.0). | [![](https://images.microbadger.com/badges/image/bytesmith/tensorflow101:release-0.2.0.svg)](https://microbadger.com/images/bytesmith/tensorflow101:release-0.2.0)
[![](https://images.microbadger.com/badges/version/bytesmith/tensorflow101:release-0.1.0.svg)](https://hub.docker.com/r/bytesmith/tensorflow101) | Contains the first set of demos and exercises, used at the [Softwerkskammer Jena](https://www.meetup.com/de-DE/jenadevs/events/255559364/) Meetup. - [GitHub Release](https://github.com/SaschaDittmann/TensorFlow101/releases/tag/0.1.0). | [![](https://images.microbadger.com/badges/image/bytesmith/tensorflow101:release-0.1.0.svg)](https://microbadger.com/images/bytesmith/tensorflow101:release-0.1.0)

## Quickstart

Insert your Azure Subscription ID in the following Docker command and start your container.

```bash
docker run -it --rm \
    -p 8888:8888 -p 6006:6006 \
    -e "SUBSCRIPTION_ID=<subscription id>" \
    bytesmith/tensorflow101
```

Use your internet browser to open the juypter notebooks.

The URL is shown in the docker logs output, e.g. `http://localhost:8888/?token=a84394577692cc3957c2b39f46f6aab02ff75b36341f8785`

The workshop notebooks are located in the *TensorFlow101* folder.

## About the Workshop

Learn how to build intelligent algorithms into apps and websites. Use popular machine learning frameworks like [Tensorflow](https://www.tensorflow.org/), [Keras](https://keras.io/), [Scikit-learn](https://scikit-learn.org/) or any other Python-based framework to build your machine learning model and train it locally or in the cloud.

The workshop starts with the machine learning basics and uses TensorFlow to explain the technical concepts behind algorithms like regressions, classifications and neural networks.

With the help of the [Azure Machine Learning service](https://docs.microsoft.com/en-us/azure/machine-learning/service/) it moves long running tasks to the cloud to build, train, and deploy your models without blocking you the resources of your local machine.

## What is Azure Machine Learning service

The [Azure Machine Learning service](https://docs.microsoft.com/en-us/azure/machine-learning/service/) is a cloud service that you can use to develop and deploy machine learning models. Using the Azure Machine Learning service, you can track your models as you build, train, deploy, and manage them, all at the broad scale that the cloud provides.
