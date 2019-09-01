#!/bin/bash

FX="np.arctan(x)"
python ../fx_gen.py --dsout datasets/example6_train.csv --fx "$FX" --rbegin -5.0 --rend 5.0 --rstep 0.01
python ../fx_fit.py --trainds datasets/example6_train.csv --modelout models/example6 \
  --epochs 500 --batch_size 100 --learning_rate 0.01 \
  --hlayers 2 --hunits 100 --hactivation relu \
  --optimizer SGD --momentum 0.0 --nesterov True


python ../fx_gen.py --dsout datasets/example6_test.csv  --fx "$FX" --rbegin -5.0 --rend 5.0 --rstep 0.0475
python ../fx_predict.py --model models/example6 --testds datasets/example6_test.csv --predicted predictions/example6_pred.csv

python ../fx_plot.py --trainds datasets/example6_train.csv --predicted predictions/example6_pred.csv
