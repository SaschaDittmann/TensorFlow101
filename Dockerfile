FROM tensorflow/tensorflow:2.4.1-jupyter
LABEL maintainer="info@bytesmith.de"
ENV SUBSCRIPTION_ID='<subscription-id>'
RUN pip install --upgrade prompt-toolkit==3.0.13 azure-cli azureml-sdk[automl,contrib,notebooks,tensorboard]==1.23.0
RUN jupyter nbextension install --py --user azureml.widgets && jupyter nbextension enable --py --user azureml.widgets
ADD notebooks /tf/TensorFlow101
