#!/bin/bash
for arg in "UCB1" "BUCB1" "BUCB2" "NormalSampling"; do
    python random_tree.py "$arg"
done