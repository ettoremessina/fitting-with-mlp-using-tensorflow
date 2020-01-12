#!/bin/bash

#Hypocycloid 8/3

FXT="(8 - 3) * np.cos(t) + 3 * np.cos((8 - 3) * t / 3)"
FYT="(8 - 3) * np.sin(t) - 3 * np.sin((8 - 3) * t / 3)"

python ../pmc2t_gen.py --dsout datasets/example4_1_train.csv --xt "$FXT" --yt "$FYT" --rbegin 0 --rend 18.54 --rstep 0.01
python ../pmc2t_fit_1.py --trainds datasets/example4_1_train.csv --modelout models/example4_1 \
  --hlayers 180 180 --hactivation tanh tanh \
  --epochs 400 --batch_size 100

python ../pmc2t_gen.py --dsout datasets/example4_1_test.csv  --xt "$FXT" --yt "$FYT" --rbegin 0 --rend 18.54 --rstep 0.0475
python ../pmc2t_predict_1.py --model models/example4_1 --testds datasets/example4_1_test.csv --predicted predictions/example4_1_pred.csv

python ../pmc2t_plot.py --ds datasets/example4_1_test.csv --predicted predictions/example4_1_pred.csv
#python ../pmc2t_plot.py --trainds datasets/example4_1_test.csv --predicted predictions/example4_1_test.csv --savefig predictions/example4_1.png
