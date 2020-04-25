#!/bin/sh

FX="np.tanh(x)"
RB=-5.0
RE=5.0

python ../fx_gen.py --dsout datasets/example8_train.csv --fx "$FX" --rbegin $RB --rend $RE --rstep 0.01
python ../fx_fit.py \
  --trainds datasets/example8_train.csv \
  --modelout models/example8 \
  --epochs 750 \
  --batch_size 150 \
  --optimizer 'Adam(learning_rate=0.01)' \
  --loss 'MeanAbsoluteError()'

python ../fx_gen.py --dsout datasets/example8_test.csv  --fx "$FX" --rbegin $RB --rend $RE --rstep 0.0475
python ../fx_predict.py --model models/example8 --ds datasets/example8_test.csv --predictionout predictions/example8_pred.csv

python ../fx_scatter.py --ds datasets/example8_test.csv --prediction predictions/example8_pred.csv
