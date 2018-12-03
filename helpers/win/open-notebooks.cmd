@echo off

docker logs tensorflow-101 |^
FOR /f "skip=1 tokens=3 Delims=:" %%A in (
	'findstr /R ":8888/?token=[0-9a-z]."'
) Do @start "" http://localhost:%%A
