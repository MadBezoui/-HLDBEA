#!/bin/bash
# Script to run all HLDBEA final benchmarks in the background using Conda 3.11 with UNBUFFERED output
cd /Users/madanibezoui/Documents/Projects/HLDBEA

echo "Starting benchmark with Python 3.11 core (Dependencies resolved!)..." > output_final_experiments.log

# Run the final benchmark using -u to force stdout to stream immediately to the log
/Users/madanibezoui/.julia/conda/3/envs/hldbea_311/bin/python -u nibea_test.py --profile configs/profile_final.yaml --params configs/params_standard.yaml >> output_final_experiments.log 2>&1

echo "Benchmark finished." >> output_final_experiments.log
