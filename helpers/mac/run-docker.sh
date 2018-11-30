#!/bin/bash
read -p 'Azure Subscription ID: ' subscription_id
docker run -d --rm -p 8888:8888 -p 6006:6006 -e "SUBSCRIPTION_ID=$subscription_id" --name tensorflow-101 -t bytesmith/tensorflow101
