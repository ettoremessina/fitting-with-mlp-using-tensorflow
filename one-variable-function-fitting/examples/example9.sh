#!/bin/bash

FX="np.sin(2 * x) / np.exp(x / 5.0)"
RB=-20.0
RE=20.0

python ../fx_gen.py --dsout datasets/example9_train.csv --fx "$FX" --rbegin $RB --rend $RE --rstep 0.01
python ../fx_fit.py --trainds datasets/example9_train.csv --modelout models/example9 \
  --hlayers 200 300 200 --hactivation sigmoid sigmoid sigmoid \
  --epochs 1000 --batch_size 200 \
  --optimizer 'Adamax()' --learning_rate 0.02

python ../fx_gen.py --dsout datasets/example9_test.csv  --fx "$FX" --rbegin $RB --rend $RE --rstep 0.0475
python ../fx_predict.py --model models/example9 --testds datasets/example9_test.csv --predictedout predictions/example9_pred.csv

python ../fx_plot.py --ds datasets/example9_test.csv --predicted predictions/example9_pred.csv
