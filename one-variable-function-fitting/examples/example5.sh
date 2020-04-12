#!/bin/bash

FX="np.log(1+np.abs(x))"
RB=-5.0
RE=5.0

python ../fx_gen.py --dsout datasets/example5_train.csv --fx "$FX" --rbegin $RB --rend $RE --rstep 0.01
python ../fx_fit.py --trainds datasets/example5_train.csv --modelout models/example5 \
  --hlayers 200 200 --hactivation relu relu \
  --epochs 500 --batch_size 100 \
  --optimizer 'RMSprop(learning_rate=0.01, rho=0.95)'

python ../fx_gen.py --dsout datasets/example5_test.csv  --fx "$FX" --rbegin $RB --rend $RE --rstep 0.0475
python ../fx_predict.py --model models/example5 --ds datasets/example5_test.csv --prediction predictions/example5_pred.csv

python ../fx_scatter.py --ds datasets/example5_test.csv --prediction predictions/example5_pred.csv
