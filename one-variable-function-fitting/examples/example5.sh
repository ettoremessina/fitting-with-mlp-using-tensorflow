#!/bin/bash

FX="np.log(1+np.abs(x))"
python ../fx_gen.py --dsout datasets/example5_train.csv --fx "$FX" --rbegin -5.0 --rend 5.0 --rstep 0.01
python ../fx_fit.py --trainds datasets/example5_train.csv --modelout models/example5 \
  --epochs 500 --batch_size 100 --learning_rate 0.01 \
  --hlayers 2 --hunits 200 --hactivation relu \
  --optimizer RMSprop --rho 0.95

python ../fx_gen.py --dsout datasets/example5_test.csv  --fx "$FX" --rbegin -5.0 --rend 5.0 --rstep 0.0475
python ../fx_predict.py --model models/example5 --testds datasets/example5_test.csv --predicted predictions/example5_pred.csv

python ../fx_plot.py --trainds datasets/example5_train.csv --predicted predictions/example5_pred.csv
