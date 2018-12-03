@echo off
docker exec -d tensorflow-101 tensorboard --logdir /notebooks/tb_data
start "" http://localhost:6006
