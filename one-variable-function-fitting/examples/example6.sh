#!/bin/bash

FX="np.arctan(x)"
RB=-5.0
RE=5.0

python ../fx_gen.py --dsout datasets/example6_train.csv --fx "$FX" --rbegin $RB --rend $RE --rstep 0.01
python ../fx_fit.py --trainds datasets/example6_train.csv --modelout models/example6 \
  --hlayers 100 150 --hactivation tanh relu \
  --epochs 500 --batch_size 100 \
  --optimizer 'SGD(learning_rate=1e-2, momentum=0.0, nesterov=True)'


python ../fx_gen.py --dsout datasets/example6_test.csv  --fx "$FX" --rbegin $RB --rend $RE --rstep 0.0475
python ../fx_predict.py --model models/example6 --ds datasets/example6_test.csv --prediction predictions/example6_pred.csv

python ../fx_plot.py --ds datasets/example6_test.csv --prediction predictions/example6_pred.csv
