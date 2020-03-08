#!/bin/bash

rm -rf dumps/example2
rm -rf logs/example2
rm -rf snaps/example2

FX="np.sin(x)"
RB=-6.0
RE=6.0
python ../fx_gen.py --dsout datasets/example2_train.csv --fx "$FX" --rbegin $RB --rend $RE --rstep 0.01
python ../fx_fit.py --trainds datasets/example2_train.csv --modelout models/example2 \
  --epochs 60 --batch_size 100  \
  --hlayers 150 150 --hactivations tanh tanh \
  --optimizer 'Adam(learning_rate=0.05, epsilon=1e-07)' \
  --dumpout dumps/example2 \
  --logsout logs/example2 \
  --modelsnapout snaps/example2 \
  --modelsnapfreq 1

python ../fx_gen.py --dsout datasets/example2_test.csv  --fx "$FX" --rbegin $RB --rend $RE --rstep 0.0475
python ../fx_predict.py --model models/example2 --ds datasets/example2_test.csv --predictionout predictions/example2_pred.csv

python ../fx_plot.py --ds datasets/example2_test.csv --prediction predictions/example2_pred.csv
#python ../fx_plot.py --ds datasets/example2_test.csv --prediction predictions/example2_pred.csv --savefig predictions/example2.png

python ../fx_video.py --modelsnap snaps/example2 --ds datasets/example2_test.csv --savevideo predictions/example2_test.gif

#ffmpeg -f gif -i example2_test2.gif example2_test2.mp4

