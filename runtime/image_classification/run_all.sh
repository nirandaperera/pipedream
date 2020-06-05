#!/bin/bash

#MODEL="vgg16_1"
CONF="4_straight"

for MODEL in vgg16_1 resnet50_1; do 
	echo "################### $MODEL $CONF"

	echo "running pipedream start"
	./run.sh "$MODEL" "$CONF" mp_conf 4
	echo "running pipedream end!"
	
	echo "running dp start"
	./run.sh "$MODEL" "$CONF" dp_conf 4
	echo "running dp end!"
	
	echo "running seq start"
	./run.sh "$MODEL" "$CONF" 1_conf 1
	echo "running seq end!"
done
