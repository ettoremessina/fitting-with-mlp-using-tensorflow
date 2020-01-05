#!/bin/bash

FX="np.sin(x)"
RB=-6.0
RE=6.0
python ../fx_gen.py --dsout datasets/example2_train.csv --fx "$FX" --rbegin $RB --rend $RE --rstep 0.01
python ../fx_fit.py --trainds datasets/example2_train.csv --modelout models/example2 \
  --epochs 250 --batch_size 100  \
  --optimizer 'Adam(epsilon=1e-07)' --learning_rate 0.05

python ../fx_gen.py --dsout datasets/example2_test.csv  --fx "$FX" --rbegin $RB --rend $RE --rstep 0.0475
python ../fx_predict.py --model models/example2 --testds datasets/example2_test.csv --predictedout predictions/example2_pred.csv

python ../fx_plot.py --ds datasets/example2_test.csv --predicted predictions/example2_pred.csv
#python ../fx_plot.py --ds datasets/example2_test.csv --predicted predictions/example2_pred.csv --savefig predictions/example2.png
