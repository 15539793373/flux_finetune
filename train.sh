#!/bin/bash
time=$(date "+%Y%m%d_%H%M%S")
CONFIG=configs/test.yaml
# 执行训练脚本并输出日志
nohup python -u train.py --config $CONFIG > ${time}.log 2>&1 &