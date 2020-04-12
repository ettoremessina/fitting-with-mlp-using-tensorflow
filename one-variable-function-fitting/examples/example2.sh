#!/bin/bash

rm -rf dumps/example2_best
rm -rf dumps/example2_recent
rm -rf logs/example2_best
rm -rf logs/example2_recent
rm -rf snaps/example2_best
rm -rf snaps/example2_recent

FX="np.sin(x)"
RB=-6.0
RE=6.0
python ../fx_gen.py --dsout datasets/example2_train.csv --fx "$FX" --rbegin $RB --rend $RE --rstep 0.01
python ../fx_gen.py --dsout datasets/example2_val.csv   --fx "$FX" --rbegin $RB --rend $RE --rstep 0.0875

python ../fx_fit.py \
  --trainds datasets/example2_train.csv \
  --valds datasets/example2_val.csv \
  --modelout models/example2_best \
  --bestmodelmonitor 'loss' \
  --epochs 60 --batch_size 100  \
  --hlayers 150 150 --hactivations tanh tanh \
  --optimizer 'Adam(learning_rate=0.05, epsilon=1e-07)' \
  --dumpout dumps/example2_best \
  --logsout logs/example2_best \
  --modelsnapout snaps/example2_best \
  --modelsnapfreq 1

  python ../fx_fit.py \
    --trainds datasets/example2_train.csv \
    --valds datasets/example2_val.csv \
    --modelout models/example2_recent \
    --epochs 60 --batch_size 100  \
    --hlayers 150 150 --hactivations tanh tanh \
    --optimizer 'Adam(learning_rate=0.05, epsilon=1e-07)' \
    --dumpout dumps/example2_recent \
    --logsout logs/example2_recent \
    --modelsnapout snaps/example2_recent \
    --modelsnapfreq 1 \
    --metrics 'accuracy'

python ../fx_gen.py --dsout datasets/example2_test.csv  --fx "$FX" --rbegin $RB --rend $RE --rstep 0.0475
python ../fx_predict.py --model models/example2_best  --ds datasets/example2_test.csv --predictionout predictions/example2_pred_best.csv
python ../fx_predict.py --model models/example2_recent --ds datasets/example2_test.csv --predictionout predictions/example2_pred_recent.csv

python ../fx_scatter.py --ds datasets/example2_test.csv --prediction predictions/example2_pred_best.csv    --title "best model"
python ../fx_scatter.py --ds datasets/example2_test.csv --prediction predictions/example2_pred_recent.csv  --title "most recent model"

python ../fx_scatter.py --ds datasets/example2_test.csv --prediction predictions/example2_pred_best.csv   --title "best model"        --savefig media/example2_best.png
python ../fx_scatter.py --ds datasets/example2_test.csv --prediction predictions/example2_pred_recent.csv --title "most recent model" --savefig media/example2_recent.png

#python ../fx_video.py --modelsnap snaps/example2_best   --ds datasets/example2_test.csv --savevideo media/example2_test_best.gif
#python ../fx_video.py --modelsnap snaps/example2_recent --ds datasets/example2_test.csv --savevideo media/example2_test_recent.gif

#ffmpeg -f gif -i media/example2_test2_best.gif   media/example2_test2_best.mp4
#ffmpeg -f gif -i media/example2_test2_recent.gif media/example2_test2_recent.mp4
