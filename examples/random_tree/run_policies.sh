#!/bin/bash

# Make 100 random trees with random state 42
python generate_random_trees.py 100 42

for policy in "BUCB1" "BUCB2" "NormalSampling"; do
    python random_tree.py 200 10 $policy
done
