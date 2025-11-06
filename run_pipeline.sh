#!/bin/bash

set -e

python audio_encoding.py
python extract_lsm_sequences.py
python train_ctc.py
