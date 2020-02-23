#!/bin/bash

FXY="y - x**2 -1"
RXB=-5.0
RXE=5.0
RYB=-5.0
RYE=5.0

python ../fxy_gen.py --dsout datasets/example3_train.csv --fxy "$FXY" --rxbegin $RXB --rxend $RXE --rybegin $RYB --ryend $RYE --rstep 0.075
python ../fxy_fit.py --trainds datasets/example3_train.csv --modelout models/example3 \
  --hlayers 200 300 200 --hactivations tanh tanh tanh \
  --epochs 20 --batch_size 100 \
  --optimizer 'Adamax(lr=0.01)' \
  --loss 'MeanSquaredError()'

python ../fxy_gen.py --dsout datasets/example3_test.csv  --fxy "$FXY" --rxbegin $RXB --rxend $RXE --rybegin $RYB --ryend $RYE  --rstep 0.275
python ../fxy_predict.py --model models/example3 --ds datasets/example3_test.csv --prediction predictions/example3_pred.csv

python ../fxy_plot.py --ds datasets/example3_test.csv --prediction predictions/example3_pred.csv
#python ../fxy_plot.py --ds datasets/example3_test.csv --prediction predictions/example3_pred.csv --savefig predictions/example3.png
