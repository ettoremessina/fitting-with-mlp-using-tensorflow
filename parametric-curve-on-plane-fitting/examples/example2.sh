#!/bin/bash

#Bernoulli's spiral

FXT="np.exp(0.1 * t) * np.cos(t)"
FYT="np.exp(0.1 * t) * np.sin(t)"

python ../pmc2t_gen.py --dsout datasets/example2_train.csv --xt "$FXT" --yt "$FYT" --rbegin 0 --rend 20.0 --rstep 0.01
python ../pmc2t_fit.py --trainds datasets/example2_train.csv --modelout models/example2 \
  --hlayers 200 200 200 \
  --hactivation sigmoid sigmoid sigmoid \
  --epochs 500 \
  --optimizer 'Adamax()'

python ../pmc2t_gen.py --dsout datasets/example2_test.csv  --xt "$FXT" --yt "$FYT" --rbegin 0 --rend 20.0 --rstep 0.0475
python ../pmc2t_predict.py --model models/example2 --ds datasets/example2_test.csv --predictionout predictions/example2_pred.csv

python ../pmc2t_plot.py --ds datasets/example2_test.csv --prediction predictions/example2_pred.csv
#python ../pmc2t_plot.py --ds datasets/example2_train.csv --prediction predictions/example2_pred.csv --savefig predictions/example2.png
