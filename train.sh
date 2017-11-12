#!/bin/bash

set -uxe

python train.py 2>&1 | tee train.log
python draw_angle_overlay.py
python draw_layers.py
