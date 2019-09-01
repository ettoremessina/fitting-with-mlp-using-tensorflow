#!/bin/bash

#Lissajous

FXT="2 * np.sin(0.5 * t + 1)"
FYT="3 * np.sin(t)"

python ../pmc2t_gen.py --dsout datasets/example5_train.csv --xt "$FXT" --yt "$FYT" --rbegin 0 --rend 12.56 --rstep 0.01
python ../pmc2t_fit.py --trainds datasets/example5_train.csv --modelout models/example5 \
  --hlayers 2 --hunits 100 --hactivation tanh \
  --epochs 1000
  #--epochs 500 --batch_size 100 --learning_rate 0.01 \
  #--hlayers 2 --hunits 200 --hactivation relu \
  #--optimizer Adam --epsilon 1e-7 \

python ../pmc2t_gen.py --dsout datasets/example5_test.csv --xt "$FXT" --yt "$FYT" --rbegin 0 --rend 12.56 --rstep 0.0475
python ../pmc2t_predict.py --model models/example5 --testds datasets/example5_test.csv --predicted predictions/example5_pred.csv

python ../pmc2t_plot.py --trainds datasets/example5_train.csv --predicted predictions/example5_pred.csv
#python ../pmc2t_plot.py --trainds datasets/example5_train.csv --predicted predictions/example5_pred.csv --savefig predictions/example5.png
