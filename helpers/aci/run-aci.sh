#!/bin/bash
AZ_SUBSCRIPTION_ID_DEFAULT=
AZ_RESOURCE_GROUP_DEFAULT=tensorflow101
ACI_NAME_DEFAULT=tensorflow-101-environment
ACI_DNS_DEFAULT=
AZ_REGION_DEFAULT=westeurope

read -p "Azure Subscription ID [$AZ_SUBSCRIPTION_ID_DEFAULT]: " AZ_SUBSCRIPTION_ID
AZ_SUBSCRIPTION_ID=${AZ_SUBSCRIPTION_ID:-$AZ_SUBSCRIPTION_ID_DEFAULT}

read -p "Azure Resource Group [$AZ_RESOURCE_GROUP_DEFAULT]: " AZ_RESOURCE_GROUP
AZ_RESOURCE_GROUP=${AZ_RESOURCE_GROUP:-$AZ_RESOURCE_GROUP_DEFAULT}

read -p "Azure Region [$AZ_REGION_DEFAULT]: " AZ_REGION
AZ_REGION=${AZ_REGION:-$AZ_REGION_DEFAULT}

read -p "Azure Container Instance Name [$ACI_NAME_DEFAULT]: " ACI_NAME
ACI_NAME=${ACI_NAME:-$ACI_NAME_DEFAULT}

read -p "Azure Container Instance DNS Prefix [$ACI_DNS_DEFAULT]: " ACI_DNS
ACI_DNS=${ACI_DNS:-$ACI_DNS_DEFAULT}

echo "Creating Resource Group..."
az group create -n $AZ_RESOURCE_GROUP -l $AZ_REGION --subscription $AZ_SUBSCRIPTION_ID

echo "Creating Azure Container Instance..."
az container create \
    --resource-group $AZ_RESOURCE_GROUP \
    --name $ACI_NAME \
    --image bytesmith/tensorflow101 \
    --memory 3 \
    --dns-name-label $ACI_DNS \
    --ports 8888 6006 \
    --environment-variables "SUBSCRIPTION_ID"="$AZ_SUBSCRIPTION_ID" \
        "RESOURCE_GROUP"="$AZ_RESOURCE_GROUP" \
        "WORKSPACE_REGION"="$AZ_REGION" \
    --subscription $AZ_SUBSCRIPTION_ID
