#!/bin/bash
for i in {1..5}; do
    for policy in "UCB1" "BUCB1" "BUCB2" "NormalSampling"; do
        python random_tree.py "$policy" "$i"
    done
done