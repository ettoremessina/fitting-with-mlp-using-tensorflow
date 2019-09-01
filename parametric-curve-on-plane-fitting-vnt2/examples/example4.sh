#!/bin/bash

#Hypocycloid 8/3

FXT="(8 - 3) * np.cos(t) + 3 * np.cos((8 - 3) * t / 3)"
FYT="(8 - 3) * np.sin(t) - 3 * np.sin((8 - 3) * t / 3)"

python ../pmc2t_gen.py --dsout datasets/example4_train.csv --xt "$FXT" --yt "$FYT" --rbegin 0 --rend 18.54 --rstep 0.01
python ../pmc2t_fit.py --trainds datasets/example4_train.csv --modelout models/example4 \
  --hlayers 2 --hunits 80 --hactivation sigmoid \
  --epochs 400
  #--epochs 500 --batch_size 100 --learning_rate 0.01 \
  #--hlayers 2 --hunits 200 --hactivation relu \
  #--optimizer Adam --epsilon 1e-7 \

python ../pmc2t_gen.py --dsout datasets/example4_test.csv  --xt "$FXT" --yt "$FYT" --rbegin 0 --rend 18.54 --rstep 0.0475
python ../pmc2t_predict.py --model models/example4 --testds datasets/example4_test.csv --predicted predictions/example4_pred.csv

python ../pmc2t_plot.py --trainds datasets/example4_train.csv --predicted predictions/example4_pred.csv
#python ../pmc2t_plot.py --trainds datasets/example4_train.csv --predicted predictions/example4_pred.csv --savefig predictions/example4.png
