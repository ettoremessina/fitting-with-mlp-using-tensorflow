#!/bin/bash

FX="np.tanh(x)"
RB=-5.0
RE=5.0

python ../fx_gen.py --dsout datasets/example8_train.csv --fx "$FX" --rbegin $RB --rend $RE --rstep 0.01
python ../fx_fit.py --trainds datasets/example8_train.csv --modelout models/example8 \
  --epochs 750 --batch_size 150 --learning_rate 0.01 \
  --loss 'MeanAbsoluteError()'

python ../fx_gen.py --dsout datasets/example8_test.csv  --fx "$FX" --rbegin $RB --rend $RE --rstep 0.0475
python ../fx_predict.py --model models/example8 --testds datasets/example8_test.csv --predictedout predictions/example8_pred.csv

python ../fx_plot.py --ds datasets/example8_test.csv --predicted predictions/example8_pred.csv
