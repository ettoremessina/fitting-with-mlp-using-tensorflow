#!/bin/sh

rm -rf snaps/example3

FX="np.exp(x)"
RB=-5.0
RE=5.0
python ../fx_gen.py --dsout datasets/example3_train.csv --fx "$FX" --rbegin $RB --rend $RE --rstep 0.01
python ../fx_fit.py --trainds datasets/example3_train.csv --modelout models/example3 \
  --hlayers 200 200 --hactivation relu relu \
  --winitializers "RandomUniform(minval=-0.1, maxval=0.1)" "TruncatedNormal(mean=0.0, stddev=0.2)" \
  --binitializers "Ones()" "Ones()" \
  --epochs 150 --batch_size 100 \
  --modelsnapout snaps/example3 \
  --modelsnapfreq 1

python ../fx_gen.py --dsout datasets/example3_test.csv  --fx "$FX" --rbegin $RB --rend $RE --rstep 0.0475
python ../fx_predict.py --model models/example3 --ds datasets/example3_test.csv --prediction predictions/example3_pred.csv

python ../fx_scatter.py --ds datasets/example3_test.csv --prediction predictions/example3_pred.csv
#python ../fx_scatter.py --ds datasets/example3_test.csv --prediction predictions/example3_pred.csv --savefig media/example3.png

python ../fx_video.py --modelsnap snaps/example3   --ds datasets/example3_test.csv --savevideo media/example3_test.gif --xlabel "x" --ylabel "y=e^x"


#ffmpeg -f gif -i media/example3_test.gif   media/example3_test.mp4
