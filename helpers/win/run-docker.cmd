@echo off

set /p AZ_SUBSCRIPTION_ID="Subscription ID: "

docker run -d --rm^
	-p 8888:8888^
	-p 6006:6006^
	-e "SUBSCRIPTION_ID=%AZ_SUBSCRIPTION_ID%"^
	--name tensorflow-101^
	-t bytesmith/tensorflow101
