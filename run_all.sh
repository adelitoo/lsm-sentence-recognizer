#!/bin/bash

set -e

python extract_lsm_traces.py --multiplier 2.1 --leak 0.002

python train_ctc_traces_linear.py
