#!/bin/bash

#Bernoulli's spiral

FXT="np.exp(0.1 * t) * np.cos(t)"
FYT="np.exp(0.1 * t) * np.sin(t)"

python ../pmc2t_gen.py --dsout datasets/example2_twin_train.csv --xt "$FXT" --yt "$FYT" --rbegin 0 --rend 20.0 --rstep 0.01
python ../pmc2t_fit_twin.py --trainds datasets/example2_twin_train.csv --modelout models/example2_twin \
  --hlayers 200 200 200 --hactivation sigmoid sigmoid sigmoid \
  --epochs 500 \
  --optimizer 'Adamax()'

python ../pmc2t_gen.py --dsout datasets/example2_twin_test.csv  --xt "$FXT" --yt "$FYT" --rbegin 0 --rend 20.0 --rstep 0.0475
python ../pmc2t_predict_twin.py --model models/example2_twin --ds datasets/example2_twin_test.csv --predictionout predictions/example2_twin_pred.csv

python ../pmc2t_plot.py --ds datasets/example2_twin_test.csv --prediction predictions/example2_twin_pred.csv
#python ../pmc2t_plot.py --ds datasets/example2_twin_train.csv --prediction predictions/example2_twin_pred.csv --savefig predictions/example2_twin.png
