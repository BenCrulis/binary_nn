#!/bin/bash

N_PARALLEL_DEFAULT=8
N_PARALLEL=$N_PARALLEL_DEFAULT

DEFAULT_DATASET=MNIST
DATASET=$DEFAULT_DATASET
N_EPOCHS=100
NUM_REPEATS=10

while [[ $# -gt 0 ]] ; do
    echo $1
    case $1 in
        --help)
            echo "use --dataset to specify a dataset for the experiments. One in {MNIST, FASHION_MNIST, CIFAR10}. Default to $DEFAULT_DATASET"
            echo "use -p or --processes to specify how many processes are run in parallel, default to $N_PARALLEL_DEFAULT"
            exit 0 ;;
        --dataset)
            shift
            DATASET=$1
            shift ;;
        -p|--processes)
            shift
            N_PARALLEL=$1
            shift ;;
        *)
            echo "unknown argument $1"
            exit 1 ;;
    esac
done

DATASET="${DATASET^^}"

EXECUTABLE="python -m binary_nn.train"

COMMON_ARGS="--dataset $DATASET -v -b 128 --epochs $N_EPOCHS --hidden 700,500,300,200 --weight-clipping 1 --run-name auto --wandb"

SEEDS=$( seq 2 $(( NUM_REPEATS+1 )) )
SEEDS=( $SEEDS )

echo "seeds: " "${SEEDS[@]}"

ARGS="--method DFA  --init 0.001 --lr 0.0001 --hidden-act signsat
--method DRTP  --init 0.001 --lr 0.0001 --hidden-act signsat
--method GD  --init 0.1 --lr 0.0001 --hidden-act signsat
--method DFA  --init 0.001 --lr 0.0001 --hidden-act tanh
--method DRTP  --init 0.001 --lr 0.0001 --hidden-act tanh
--method GD  --init 0.1 --lr 0.0001 --hidden-act tanh
--method DFA --binarize --init 0.001 --lr 0.000001 --hidden-act signsat
--method DRTP --binarize --init 0.001 --lr 0.00001 --hidden-act signsat
--method GD --binarize --init 0.1 --lr 0.00001 --hidden-act signsat
--method DFA --binarize --init 0.1 --lr 0.0001 --hidden-act tanh
--method DRTP --binarize --init 0.001 --lr 0.00001 --hidden-act tanh
--method GD --binarize --init 0.1 --lr 0.00001 --hidden-act tanh"

IFS=$'\n' ARGS=( $ARGS )

echo "first arg:" $ARGS

n=${#ARGS[@]}

N=$(( n*$NUM_REPEATS ))

echo "$n commands to execute $NUM_REPEATS times each: $N processes"

echo "starting experiments"

i=0

while [[ i -lt $N ]]
do
  current_jobs=$(jobs -pr)
  running=0
  for job in $current_jobs
  do
    let running++
  done

  to_run=$((N_PARALLEL - running))

  for j in $(seq 0 $((to_run - 1)))
  do
    ARG_I=$(( i/NUM_REPEATS ))
    SEED_I=$(( i%NUM_REPEATS ))
    ARG=${ARGS[$ARG_I]}
    SEED=${SEEDS[$SEED_I]}
    echo "ARG: $ARG, SEED: $SEED"
    echo "running command number $i"
    #eval $cmd &
    FULLCOMMAND="$EXECUTABLE $COMMON_ARGS $ARG --seed $SEED"
    echo "running job with command: $FULLCOMMAND"
    eval $FULLCOMMAND &>> output.log &
    let i++
  done
  sleep 1
done

wait

echo "done"
