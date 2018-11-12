#!/bin/bash
read -p 'Azure Subscription ID: ' subscription_id

#login to azure using your credentials
az account show 1> /dev/null

if [ $? != 0 ];
then
	az login
fi

az account set --subscription $subscription_id

az provider show -n Microsoft.ContainerInstance -o table
az provider register -n Microsoft.ContainerInstance

export SUBSCRIPTION_ID=$subscription_id
python ./scripts/00_Setup.py