#!/bin/bash

export FLASK_APP=./app.py 
CUDA_VISIBLE_DEVICES="0" flask run -p 5000
