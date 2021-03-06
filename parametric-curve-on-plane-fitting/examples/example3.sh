#!/bin/bash

#Hypocycloid 5/1

FXT="(5 - 1) * np.cos(t) + 1 * np.cos((5 - 1) * t / 1)"
FYT="(5 - 1) * np.sin(t) - 1 * np.sin((5 - 1) * t / 1)"

python ../pmc2t_gen.py --dsout datasets/example3_train.csv --xt "$FXT" --yt "$FYT" --rbegin 0 --rend 6.28 --rstep 0.01
python ../pmc2t_fit.py --trainds datasets/example3_train.csv --modelout models/example3 \
  --hlayers 100 200 100 --hactivation hard_sigmoid hard_sigmoid hard_sigmoid \
  --epochs 500

python ../pmc2t_gen.py --dsout datasets/example3_test.csv  --xt "$FXT" --yt "$FYT" --rbegin 0 --rend 6.28 --rstep 0.0475
python ../pmc2t_predict.py --model models/example3 --ds datasets/example3_test.csv --predictionout predictions/example3_pred.csv

python ../pmc2t_plot.py --ds datasets/example3_test.csv --prediction predictions/example3_pred.csv
#python ../pmc2t_plot.py --ds datasets/example3_test.csv --prediction predictions/example3_pred.csv --savefig predictions/example3.png
