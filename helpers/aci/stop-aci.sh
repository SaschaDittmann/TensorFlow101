#!/bin/bash
AZ_SUBSCRIPTION_ID_DEFAULT=
AZ_RESOURCE_GROUP_DEFAULT=tensorflow101
ACI_NAME_DEFAULT=tensorflow-101-environment

read -p "Azure Subscription ID [$AZ_SUBSCRIPTION_ID_DEFAULT]: " AZ_SUBSCRIPTION_ID
AZ_SUBSCRIPTION_ID=${AZ_SUBSCRIPTION_ID:-$AZ_SUBSCRIPTION_ID_DEFAULT}

read -p "Azure Resource Group [$AZ_RESOURCE_GROUP_DEFAULT]: " AZ_RESOURCE_GROUP
AZ_RESOURCE_GROUP=${AZ_RESOURCE_GROUP:-$AZ_RESOURCE_GROUP_DEFAULT}

read -p "Azure Container Instance Name [$ACI_NAME_DEFAULT]: " ACI_NAME
ACI_NAME=${ACI_NAME:-$ACI_NAME_DEFAULT}

echo "Deleting Azure Container Instance..."
az container delete \
    --resource-group $AZ_RESOURCE_GROUP \
    --name $ACI_NAME \
    --subscription $AZ_SUBSCRIPTION_ID \
    --yes
