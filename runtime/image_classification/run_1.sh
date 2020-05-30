#!/bin/bash
export CUDA_VISIBLE_DEVICES=4

LOGS_DIR=logs_1
mkdir -p $LOGS_DIR
rm -rf $LOGS_DIR/*

command() {
	echo "### batch size $1 rank $2 start"
	python main_with_runtime.py --module models.alexnet.gpus=4_straight -b "$1" --data_dir /N/u2/d/dnperera/data/imagenet-mini/ --rank "$2" --local_rank "$2" --master_addr localhost --config_path models/alexnet/gpus=4_straight/1_conf.json --distributed_backend gloo --epochs 3 -v 1
	echo "### batch size $1 rank $2 end"
}

for b in 64 128 256 512
do
	echo "batch $b start"
#	for r in 0 
#	do
	r=0
	command $b $r &> "$LOGS_DIR"/"$b"_"$r".log 
#	done
	wait
	echo "batch $b done!"
	echo "-----------------------------------------------------------"
done
