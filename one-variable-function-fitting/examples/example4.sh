#!/bin/bash

FX="np.sqrt(np.abs(x))"
RB=-5.0
RE=5.0

python ../fx_gen.py --dsout datasets/example4_train.csv --fx "$FX" --rbegin $RB --rend $RE --rstep 0.01
python ../fx_fit.py --trainds datasets/example4_train.csv --modelout models/example4 --epochs 500 --batch_size 100 --learning_rate 0.01

python ../fx_gen.py --dsout datasets/example4_test.csv  --fx "$FX" --rbegin $RB --rend $RE --rstep 0.0475
python ../fx_predict.py --model models/example4 --testds datasets/example4_test.csv --predictedout predictions/example4_pred.csv

python ../fx_plot.py --ds datasets/example4_test.csv --predicted predictions/example4_pred.csv
