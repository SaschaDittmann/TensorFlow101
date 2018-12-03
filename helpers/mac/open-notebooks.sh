#!/bin/bash
open $(docker logs tensorflow-101 | grep -e ':8888/?token=' | tail -1 | sed s/'('[a-z0-9]*' or 127.0.0.1)'/localhost/g | awk '{print $1}')
