#!/bin/bash

#Hypocycloid 5/1

FXT="(5 - 1) * np.cos(t) + 1 * np.cos((5 - 1) * t / 1)"
FYT="(5 - 1) * np.sin(t) - 1 * np.sin((5 - 1) * t / 1)"
python ../pmc2t_gen.py --dsout datasets/example3_train.csv --xt "$FXT" --yt "$FYT" --rbegin 0 --rend 6.28 --rstep 0.01
python ../pmc2t_fit.py --trainds datasets/example3_train.csv --modelout models/example3 \
  --hlayers 2 --hunits 100 --hactivation hard_sigmoid \
  --epochs 1000
  #--epochs 500 --batch_size 100 --learning_rate 0.01 \
  #--hlayers 2 --hunits 200 --hactivation relu \
  #--optimizer Adam --epsilon 1e-7 \

python ../pmc2t_gen.py --dsout datasets/example3_test.csv  --xt "$FXT" --yt "$FYT" --rbegin 0 --rend 6.28 --rstep 0.0475
python ../pmc2t_predict.py --model models/example3 --testds datasets/example3_test.csv --predicted predictions/example3_pred.csv

python ../pmc2t_plot.py --trainds datasets/example3_train.csv --predicted predictions/example3_pred.csv
#python ../pmc2t_plot.py --trainds datasets/example3_train.csv --predicted predictions/example3_pred.csv --savefig predictions/example3.png
