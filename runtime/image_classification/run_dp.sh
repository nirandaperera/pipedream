#!/bin/bash
export CUDA_VISIBLE_DEVICES=4,5,6,7

# MODULE="models.resnet50.gpus=4_straight"
MODULE="models.alexnet.gpus=4_straight"

# CONFIG="models/resnet50/gpus=4_straight/mp_conf.json"
CONFIG="models/alexnet/gpus=4_straight/dp_conf.json"

DATA="/N/u2/d/dnperera/data/imagenet-mini/"
# DATA="/N/u2/d/dnperera/data/ILSVRC/Data/CLS-LOC/"

LOGS_DIR="logs_dp/$MODULE"
echo "mking dir $LOGS_DIR"
mkdir -p "$LOGS_DIR"
rm "$LOGS_DIR"/*

command() {
        echo "### batch size $1 rank $2 start"
        python main_with_runtime.py --module "$MODULE" -b "$1" --data_dir "$DATA" --rank "$2" --local_rank "$2" --master_addr localhost --config_path "$CONFIG" --distributed_backend gloo --epochs 3 -v 1  -j 10
        echo "### batch size $1 rank $2 end"
}

for b in 64 128 256 512
do
        echo "batch $b start"
        for r in 0 1 2 3
        do
                command $b $r &> "$LOGS_DIR"/"$b"_"$r".log &
        done
        wait
        echo "batch $b done!"
        echo "-----------------------------------------------------------"
done

echo "zipping $LOGS_DIR"
cd $LOGS_DIR && tar czf ../../logs_dp_"$MODULE".tar *.log && cd - 
