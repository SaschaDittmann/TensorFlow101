#!/bin/bash
docker exec -d tensorflow-101 tensorboard --logdir /notebooks/tb_data
open 'http://localhost:6006'