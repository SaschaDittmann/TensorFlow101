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

open $(az container logs -g $AZ_RESOURCE_GROUP -n $ACI_NAME --subscription $AZ_SUBSCRIPTION_ID | grep -e ':8888/?token=' | tail -1 | sed s/'('.*' or 127.0.0.1)'/$ACI_DNS.$AZ_REGION.azurecontainer.io/g | awk '{print $1}')
