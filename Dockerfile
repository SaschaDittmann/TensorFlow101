FROM tensorflow/tensorflow:1.10.0-py3
MAINTAINER cloudblog@gmx.de
ENV SUBSCRIPTION_ID='<subscription-id>'
RUN pip install --upgrade keras azure-cli azureml-sdk[notebooks,automl]
ADD notebooks /notebooks/TensorFlow101
