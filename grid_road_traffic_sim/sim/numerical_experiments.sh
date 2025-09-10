#!/bin/bash

for i in {1..15}; do
  for j in {1..15}; do
    python run.py $i $j &
    wait
  done
done

wait

echo FINISHED!!