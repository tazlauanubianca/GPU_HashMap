#!/bin/bash
set -e

# Job parameters.
mem=512
time_per_job="4:00"
cpu="rusage[mem=${mem}]"
gpu="rusage[mem=${mem}, ngpus_excl_p=1]"
nodes=8

bsub -W ${time_per_job} -n ${nodes} -R "${gpu}" "python bench.py ${@}" 2>&1
#bsub -W ${time_per_job} -n ${nodes} -R "${gpu}" "./gpu_hashtable 10000000 1" 2>&1

