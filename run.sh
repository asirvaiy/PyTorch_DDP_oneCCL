#!/bin/bash
source $ONEAPI_INSTALL/setvars.sh --force > /dev/null 2>&1
source activate pytorch
echo "########## Executing the run"
 mpirun -n 4  -l python ccl_demo.py > cpu2.txt
echo "########## Done with the run"
