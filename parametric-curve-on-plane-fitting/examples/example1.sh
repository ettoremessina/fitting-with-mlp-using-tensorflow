#!/bin/bash

#Archimedean spiral

FXT="0.1 * t * np.cos(t)"
FYT="0.1 * t * np.sin(t)"

python ../pmc2t_gen.py --dsout datasets/example1_train.csv --xt "$FXT" --yt "$FYT" --rbegin 0 --rend 20.0 --rstep 0.01
python ../pmc2t_fit.py --trainds datasets/example1_train.csv --modelout models/example1 \
  --hlayers 200 300 200 --hactivation sigmoid tanh sigmoid \
  --epochs 250

python ../pmc2t_gen.py --dsout datasets/example1_test.csv  --xt "$FXT" --yt "$FYT" --rbegin 0 --rend 20.0 --rstep 0.0475
python ../pmc2t_predict.py --model models/example1 --ds datasets/example1_test.csv --predictionout predictions/example1_pred.csv

python ../pmc2t_plot.py --ds datasets/example1_test.csv --prediction predictions/example1_pred.csv
#python ../pmc2t_plot.py --ds datasets/example1_train.csv --prediction predictions/example1_pred.csv --savefig predictions/example1.png
