#!/bin/bash

FX="0.5*x**3 - 2*x**2 - 3*x - 1"
python ../fx_gen.py --dsout datasets/example1_train.csv --fx "$FX" --rbegin -5.0 --rend 5.0 --rstep 0.01
python ../fx_fit.py --trainds datasets/example1_train.csv --modelout models/example1 \
  --epochs 500 --batch_size 100 --learning_rate 0.01 \
  --hlayers 2 --hunits 200 --hactivation relu \
  --optimizer Adam --epsilon 1e-7 \

python ../fx_gen.py --dsout datasets/example1_test.csv  --fx "$FX" --rbegin -5.0 --rend 5.0 --rstep 0.0475
python ../fx_predict.py --model models/example1 --testds datasets/example1_test.csv --predicted predictions/example1_pred.csv

python ../fx_plot.py --trainds datasets/example1_train.csv --predicted predictions/example1_pred.csv
#python ../fx_plot.py --trainds datasets/example1_train.csv --predicted predictions/example1_pred.csv --savefig predictions/example1.png
