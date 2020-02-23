#!/bin/bash

FXY="np.sin(np.sqrt(x**2 + y**2))"
RXB=-5.0
RXE=5.0
RYB=-5.0
RYE=5.0

python ../fxy_gen.py --dsout datasets/example2_train.csv --fxy "$FXY" --rxbegin $RXB --rxend $RXE --rybegin $RYB --ryend $RYE --rstep 0.075
python ../fxy_fit.py --trainds datasets/example2_train.csv --modelout models/example2 \
  --hlayers 100 100 --hactivations relu relu \
  --epochs 10 --batch_size 50 \
  --optimizer 'SGD(decay=1e-6, momentum=0.9, nesterov=True)' \
  --loss 'MeanSquaredError()'

python ../fxy_gen.py --dsout datasets/example2_test.csv  --fxy "$FXY" --rxbegin $RXB --rxend $RXE --rybegin $RYB --ryend $RYE  --rstep 0.275
python ../fxy_predict.py --model models/example2 --ds datasets/example2_test.csv --prediction predictions/example2_pred.csv

python ../fxy_plot.py --ds datasets/example2_test.csv --prediction predictions/example2_pred.csv
#python ../fxy_plot.py --ds datasets/example2_test.csv --prediction predictions/example2_pred.csv --savefig predictions/example2.png
