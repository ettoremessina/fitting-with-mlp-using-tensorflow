#!/bin/bash

#Bernoulli's spiral

FXT="np.exp(0.1 * t) * np.cos(t)"
FYT="np.exp(0.1 * t) * np.sin(t)"
python ../pmc2t_gen.py --dsout datasets/example2_train.csv --xt "$FXT" --yt "$FYT" --rbegin 0 --rend 20.0 --rstep 0.01
python ../pmc2t_fit.py --trainds datasets/example2_train.csv --modelout models/example2 \
  --hlayers 2 --hunits 100 --hactivation hard_sigmoid \
  --epochs 1000
  #--epochs 500 --batch_size 100 --learning_rate 0.01 \
  #--hlayers 2 --hunits 200 --hactivation relu \
  #--optimizer Adam --epsilon 1e-7 \

python ../pmc2t_gen.py --dsout datasets/example2_test.csv  --xt "$FXT" --yt "$FYT" --rbegin 0 --rend 20.0 --rstep 0.0475
python ../pmc2t_predict.py --model models/example2 --testds datasets/example2_test.csv --predicted predictions/example2_pred.csv

python ../pmc2t_plot.py --trainds datasets/example2_train.csv --predicted predictions/example2_pred.csv
#python ../pmc2t_plot.py --trainds datasets/example2_train.csv --predicted predictions/example2_pred.csv --savefig predictions/example2.png
