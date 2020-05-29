#!/bin/bash

echo "running pipedream start"
./run.sh
echo "running pipedream end!"


echo "running dp start"
./run_dp.sh
echo "running dp end!"


echo "running seq start"
./run_1.sh
echo "running seq end!"
