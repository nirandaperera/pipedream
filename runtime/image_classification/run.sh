#!/bin/bash
export CUDA_VISIBLE_DEVICES=4,5,6,7

DATA="/N/u2/d/dnperera/data/imagenet-mini/"

MODEL="$1"
CONF="$2"
JSON="$3"
RANKS="$4"
BATCH_SIZE="$5"

MODULE="models.$BATCH_SIZE.$MODEL.gpus=$CONF"
CONFIG="models/$BATCH_SIZE/$MODEL/gpus=$CONF/$JSON.json"

echo "@@@ module $MODULE config $CONFIG ranks $RANKS batch size $BATCH_SIZE"

LOGS_DIR="logs/${BATCH_SIZE}_${MODEL}_${CONF}_${JSON}"
echo "making dir $LOGS_DIR"
mkdir -p "$LOGS_DIR"
rm -f "$LOGS_DIR"/*

command() {
  echo "### batch size $1 rank $2 start"
  python main_with_runtime.py --module "$MODULE" -b "$1" --data_dir $DATA --rank "$2" --local_rank "$2" \
    --master_addr localhost --config_path "$CONFIG" --distributed_backend gloo --epochs 3 -v 1 -j 10
  echo "### batch size $1 rank $2 end"
}

for ((r = 0; r < RANKS; r++)); do
  command "$BATCH_SIZE" $r &>"$LOGS_DIR"/"$BATCH_SIZE"_"$r".log &
done
wait

echo "zipping $LOGS_DIR"
cd "$LOGS_DIR" && tar czf ../../"${MODEL}_${CONF}_${JSON}".tar ./*.log && cd - || return

echo "@@@ module $MODULE config $CONFIG ranks $RANKS batch size $BATCH_SIZE DONE!"
