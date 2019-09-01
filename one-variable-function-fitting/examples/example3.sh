#!/bin/bash

FX="np.exp(x)"
python ../fx_gen.py --dsout datasets/example3_train.csv --fx "$FX" --rbegin -5.0 --rend 5.0 --rstep 0.01
python ../fx_fit.py --trainds datasets/example3_train.csv --modelout models/example3 \
  --epochs 500 --batch_size 100 --learning_rate 0.001 \
  --hlayers 2 --hunits 200 --hactivation relu


python ../fx_gen.py --dsout datasets/example3_test.csv  --fx "$FX" --rbegin -5.0 --rend 5.0 --rstep 0.0475
python ../fx_predict.py --model models/example3 --testds datasets/example3_test.csv --predicted predictions/example3_pred.csv

python ../fx_plot.py --trainds datasets/example3_train.csv --predicted predictions/example3_pred.csv
#python ../fx_plot.py --trainds datasets/example3_train.csv --predicted predictions/example3_pred.csv --savefig predictions/example3.png
