FROM tensorflow/tensorflow:1.10.0-py3
LABEL maintainer="info@bytesmith.de"
ENV SUBSCRIPTION_ID='<subscription-id>'
RUN pip install --upgrade keras==2.1.5 azure-cli azureml-sdk[automl,contrib,notebooks]==1.0.21
RUN jupyter nbextension install --py --user azureml.widgets && jupyter nbextension enable --py --user azureml.widgets
ADD notebooks /notebooks/TensorFlow101
