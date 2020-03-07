#!/bin/bash
rm -rf dumps/example1
rm -rf logs/example1

FX="0.5*x**3 - 2*x**2 - 3*x - 1"
RB=-10.0
RE=10.0

python ../fx_gen.py --dsout datasets/example1_train.csv --fx "$FX" --rbegin $RB --rend $RE --rstep 0.01
python ../fx_gen.py --dsout datasets/example1_val.csv  --fx "$FX" --rbegin $RB --rend $RE --rstep 0.0875

python ../fx_fit.py \
  --trainds datasets/example1_train.csv \
  --valds datasets/example1_val.csv \
  --modelout models/example1 \
  --hlayers 120 160 --hactivations tanh relu \
  --epochs 100 --batch_size 50 \
  --optimizer 'Adam(learning_rate=1e-2, epsilon=1e-07)' \
  --loss 'MeanSquaredError()' \
  --metrics 'mean_absolute_error' 'mean_absolute_error' 'mean_squared_logarithmic_error' 'cosine_similarity' \
  --dumpout dumps/example1 \
  --logsout logs/example1

python ../fx_gen.py --dsout datasets/example1_test.csv  --fx "$FX" --rbegin $RB --rend $RE --rstep 0.0475
python ../fx_predict.py --model models/example1 --ds datasets/example1_test.csv --prediction predictions/example1_pred.csv

python ../fx_plot.py --ds datasets/example1_test.csv --prediction predictions/example1_pred.csv
#python ../fx_plot.py --ds datasets/example1_test.csv --prediction predictions/example1_pred.csv --savefig predictions/example1.png

python ../fx_diag.py --dump dumps/example1
