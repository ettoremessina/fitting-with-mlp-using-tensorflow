# One-variable real-valued function fitting
This project implements the fitting of a continuous and limited real-valued function defined in a closed interval of the reals.
This one-variable real-valued function fitting is implemented using a configurable multilayer perceptron neural network written using TensorFlow 2 & Keras; it requires TensorFlow 2.0.0 or 2.1.0 library and also NumPy and MatPlotLib libraries.<br/>

Please visit [here](https://computationalmindset.com/en/neural-networks/one-variable-real-function-fitting-with-tensorflow.html) for concepts about this project.

It contains six python programs:
 - **fx_gen.py** generates a synthetic dataset file invoking a one-variable real-valued function on an real interval.
 - **fx_fit.py** fits a one-variable real-valued function in an interval using a configurable multilayer perceptron.
 - **fx_predict.py** makes a prediction of a one-variable real-valued function modeled with a pretrained multilayer perceptron.
 - **fx_plot.py** generates two overlapped x/y scatter graphs: the blue one is the input dataset, the red one is the prediction.
 - **fx_diag.py** generates a set of graphs to show the curves of loss function and the curves of metrics (in they are available both on training and validation datasets.
 - **fx_video.py** generates an animated gif that shows the prediction curve computed on an input dataset as the epochs change.

### Predefined examples of usage of the four command in cascade
In the subfolder **examples** there are nine shell scripts to fit nine different one-variable real-valued functions; each script executes the four programs in cascade in order to reach and show the goal.

```bash
$ cd one-variable-function-fitting/examples
$ sh example1.sh
$ sh example2.sh
$ sh example3.sh
$ sh example4.sh
$ sh example5.sh
$ sh example6.sh
$ sh example7.sh
$ sh example8.sh
$ sh example9.sh
```

For details about the four commands and their command line options, please read below.


## fx_gen.py<a name="fx_gen"/>
To get the usage of [fx_gen.py](./fx_gen.py) please run:
```bash
$ python fx_gen.py --help
```

and you get:
```
usage: fx_gen.py [-h]
                 --dsout DS_OUTPUT_FILENAME
                 --fx FUNC_X_BODY
                 [--rbegin RANGE_BEGIN]
                 [--rend RANGE_END]
                 [--rstep RANGE_STEP]

fx_gen.py generates a synthetic dataset file calling a one-variable real-valued function in an interval

optional arguments:
  -h, --help                 show this help message and exit
  --dsout DS_OUTPUT_FILENAME dataset output file (csv format)
  --fx FUNC_X_BODY           f(x) body (lamba format)
  --rbegin RANGE_BEGIN       begin range (default:-5.0)
  --rend RANGE_END           end range (default:+5.0)
  --rstep RANGE_STEP         step range (default: 0.01)
```

Namely:
- **-h or --help** shows the above usage
- **--rbegin** and **--rend** are the limit of the closed interval of reals of independent variable x.
- **--rstep** is the incremental step of independent variable x into the interval.
- **--fx** is the one-variable real-value function to use to compute the value of dependent variable; it is in lamba body format.
- **--dsout** is the target dataset file name. The content of this file is csv (no header at first line) and each line contains a pair of real numbers: the x and the f(x) where x is a value of the interval and f(x) is the value of dependent variable. This argument is mandatory.

### Examples of fx_gen.py usage
```bash
$ python fx_gen.py --dsout mydataset.csv  --fx "np.exp(np.sin(x))" --rbegin -6.0 --rend 6.0 --rstep 0.05

$ python fx_gen.py --dsout mydataset.csv  --fx "np.sqrt(np.abs(x))" --rbegin -5.0 --rend 5.0 --rstep 0.04
```


## fx_fit.py<a name="fx_fit"/>
To get the usage of [fx_fit.py](./fx_fit.py) please run:
```bash
$ python fx_fit.py --help
```

and you get:
```
usage: fx_fit.py [-h]
                 --trainds TRAIN_DATASET_FILENAME
                 --modelout MODEL_PATH
                 [--valds VAL_DATASET_FILENAME]
                 [--epochs EPOCHS] [--batch_size BATCH_SIZE]
                 [--hlayers HIDDEN_LAYERS_LAYOUT [HIDDEN_LAYERS_LAYOUT ...]]
                 [--hactivations ACTIVATION_FUNCTIONS [ACTIVATION_FUNCTIONS ...]]
                 [--optimizer OPTIMIZER]
                 [--loss LOSS]
                 [--metrics METRICS [METRICS ...]]
                 [--dumpout DUMPOUT_PATH]
                 [--logsout LOGSOUT_PATH]
                 [--modelsnapout MODEL_SNAPSHOTS_PATH]
                 [--modelsnapfreq MODEL_SNAPSHOTS_FREQ]                 

fx_fit.py fits a one-variable real-valued function dataset using a configurable multilayer perceptron network

optional arguments:
  -h, --help                        show this help message and exit
  --trainds TRAIN_DATASET_FILENAME  train dataset file (csv format)
  --valds VAL_DATASET_FILENAME      validation dataset file (csv format)
  --modelout MODEL_PATH             output model directory
  --epochs EPOCHS                   number of epochs
  --batch_size BATCH_SIZE           batch size
  --hlayers HIDDEN_LAYERS_LAYOUT [HIDDEN_LAYERS_LAYOUT ...] number of neurons for each hidden layer
  --hactivations ACTIVATION_FUNCTIONS [ACTIVATION_FUNCTIONS ...] activation functions between layer
  --optimizer OPTIMIZER             optimizer algorithm
  --loss LOSS                       loss function
  --metrics METRICS [METRICS ...]   metrics
  --dumpout DUMPOUT_PATH           dump directory (directory to store loss and metric values)
  --logsout LOGSOUT_PATH           logs directory for TensorBoard
  --modelsnapout MODEL_SNAPSHOTS_PATH  output model snapshots directory
  --modelsnapfreq MODEL_SNAPSHOTS_FREQ frequency in epochs to make the snapshot of model  
```

Namely:
- **-h or --help** shows the above usage
- **--trainds** is the input training dataset in csv format: a pair of real numbers for each line respectively for x and y (no header at first line). In case you haven't a such real world true dataset, for your experiments you can generate it synthetically using **fx_gen.py**. This argument is mandatory.
- **--valds** is the input validation dataset in csv format (same structure than train dataset) that is used by the program to compute loss function and metrics over it. This argument is optional; if it is not specified, program will not compute loss and metric on any validation dataset, but only on training dataset.
- **--modelout** is a non-existing directory where the program saves the trained model (in tf native format). This argument is mandatory.
- **--epochs** is the number of epochs of the training process. The default is **500**
- **--batch_size** is the size of the batch used during training. The default is **50**
- **--hlayers** is a sequence of integers: the size of the sequence is the number of hidden layers, each value of the sequence is the number of neurons in the correspondent layer. The default is **100** (that means one only hidden layer with 100 neurons),
- **--hactivations** is a sequence of activation function names: the size of the sequence must be equal to the number of layers and each item of the sequence is the activation function to apply to the output of the neurons of the correspondent layer; please see [TensorFlow 2 activation function reference](https://www.tensorflow.org/api_docs/python/tf/keras/activations) for details and examples at the end of this section.\
  Available activation functions are:
  - elu
  - exponential
  - hard_sigmoid
  - linear
  - relu
  - selu
  - sigmoid
  - softmax
  - softplus
  - softsign
  - tanh\
  The default is **relu** (applied to one only hidden layer; if number of layers is > 1, this argument becomes mandatory).
- **--optimizer** is the constructor call of the algorithm used by the training process. You can pass also named arguments between round brackets; please see [TensorFlow 2 optimizer reference](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers) for details about constructor named parameters and examples at the end of this section.\
  Available algorithm constructors are:
  - Adadelta()
  - Adagrad()
  - Adam()
  - Adamax()
  - Ftrl()
  - Nadam()
  - RMSprop()
  - SGD()\
  The default is **Adam()**.
- **--loss** is the constructor call of the loss function used by the training process. You can pass also named arguments between round brackets; please see [TensorFlow 2 loss functions reference](https://www.tensorflow.org/api_docs/python/tf/keras/losses) for details about constructor named parameters and examples at the end of this section.\
  Available loss function construtors are:
  - BinaryCrossentropy()
  - CategoricalCrossentropy()
  - CategoricalHinge()
  - CosineSimilarity()
  - Hinge()
  - Huber()
  - KLDivergence()
  - LogCosh()
  - MeanAbsoluteError()
  - MeanAbsolutePercentageError()
  - MeanSquaredError()
  - MeanSquaredLogarithmicError()
  - Poisson()
  - Reduction()
  - SparseCategoricalCrossentropy()
  - SquaredHinge()\
  The default is **MeanSquaredError()**.
- **--metric** is a sequence of metric names; please see [TensorFlow 2 metrics reference](https://www.tensorflow.org/api_docs/python/tf/keras/metrics/); anyway because the program is implementing a regression the more interesting metrics to use are:
  - mean_squared_error
  - mean_absolute_error
  - mean_absolute_percentage_error
  - cosine_proximity
- **--dumpout** is the directory where program stores (in different csv files) the values of loss function and metrics computed on training dataset and, when passed with **--valds** argument, on validation dataset, too. This program does not clean old content of this folder: it is advisable to delete the dumpout folder before launching this program. This argument is optional.
- **--logsout** is the directory where program stores the logs for TensorBoard; This program does not clean old content of this folder: it is advisable to delete the dumpout folder before launching this program. This argument is optional. To browse the generated log with TensorBoard pass this folder to **--logdir** argument of **tensorboard** program.
- **--modelsnapout**  is the directory where program stores periodically (see **--modelsnapfreq** argument) the snapshots of the model as the epochs change. This program does not clean old content of this folder: it is advisable to delete the modelsnapout folder before launching this program. This argument is optional.
- **--modelsnapfreq** is the frequency in epochs to save the snapshots of current model in **--modelsnapout**. This argument is optional and defaulted to 25 (that means a snapshot of model each 25 epochs); if **--modelsnapout** is not specified, this argument is ignored, otherwise model snapshots are saved in according with this argument as the epochs change and it is ensured that models at first and last epochs are saved whatever the frequency.
### Examples of fx_fix.py usage
```bash
$ python fx_fit.py \
  --trainds mytrainds.csv \
  --modelout mymodel \
  --hlayers 200 200
  --hactivation relu relu \
  --epochs 500 \
  --batch_size 100

$ python fx_fit.py \
  --trainds mytrainds.csv \
  --modelout mymodel \
  --hlayers 120 160 \
  --hactivations tanh relu \
  --epochs 100 \
  --batch_size 50 \
  --optimizer 'Adam(learning_rate 0.05, epsilon=1e-07)' \
  --loss 'MeanSquaredError()'

$ python fx_fit.py \
  --trainds mytrainds.csv \
  --modelout mymodel \
  --hlayers 200 300 200 \
  --hactivation sigmoid sigmoid sigmoid \
  --epochs 1000 \
  --batch_size 200 \
  --optimizer 'Adamax(learning_rate=0.02)

$ python fx_fit.py \
  --trainds mytrainds.csv \
  --modelout mymodel \
  --hlayers 200 300 200 \
  --hactivation sigmoid sigmoid sigmoid \
  --epochs 1000 \
  --batch_size 200 \
  --optimizer 'Adamax(learning_rate=0.02)'
```


## fx_predict.py<a name="fx_predict"/>
To get the usage of [fx_predict.py](./fx_predict.py) please run
```bash
$ python fx_predict.py --help
```

and you get:
```
usage: fx_predict.py [-h]
                     --model MODEL_PATH
                     --ds TEST_DATASET_FILENAME
                     --predictionout PREDICTION_DATA_FILENAME

fx_predict.py makes prediction of the values of a one-variable real-valued function modeled with a pretrained multilayer perceptron

optional arguments:
  -h, --help                               show this help message and exit
  --model MODEL_PATH                       model directory
  --ds DATASET_FILENAME                    input dataset file (csv format); only x-values are used
  --predictionout PREDICTION_DATA_FILENAME prediction data file (csv format)
```

Namely:
- **-h or --help** shows the above usage
- **--model** is the directory of a model generated by **fx_fit.py** (see **--modelout** command line parameter of **fx_fit.py**). This argument is mandatory.
- **--ds** is the input dataset in csv format (no header at first line): program uses only the x values (first column). In case you haven't a such real world true dataset, for your experiments you can generate it synthetically using **fx_gen.py**. This argument is mandatory.
- **--predictionout** is the file name of prediction values. The content of this file is csv (no header at first line) and each line contains a pair of real numbers: the x value comes from input dataset and the prediction is the value of f(x) computed by multilayer perceptron model on x value; this argument is mandatory.

### Example of fx_predict.py usage
```bash
$ python fx_predict.py --model mymodel --ds mytestds.csv --predictionout myprediction.csv
```


## fx_plot.py<a name="fx_plot"/>
To get the usage of [fx_plot.py](./fx_plot.py) please run
```bash
$ python fx_plot.py --help
```

and you get:
```
usage: fx_plot.py [-h]
                  --ds DATASET_FILENAME
                  --prediction PREDICTION_DATA_FILENAME
                  [--savefig SAVE_FIGURE_FILENAME]

fx_plot.py shows two overlapped x/y scatter graphs: the blue one is the dataset, the red one is the prediction one

optional arguments:
  -h, --help            show this help message and exit
  --ds DATASET_FILENAME dataset file (csv format)
  --prediction PREDICTION_DATA_FILENAME  prediction data file (csv format)
  --savefig SAVE_FIGURE_FILENAME       if present, the chart is saved on a file instead to be shown on screen
```
Namely:
- **-h or --help** shows the above usage
- **--ds** is an input dataset in csv format (no header at first line). Usually this parameter is the test dataset file passed to **fx_predict.py**, but you could pass the training dataset passed to **fx_fit.py**. This argument is mandatory.
- **--prediction** is the file name of prediction values generated by **fx_predict.py** (see **--predictionout** command line parameter of **fx_predict.py**). This argument is mandatory.
- **--savefig** if this argument is missing, the chart is shown on screen, otherwise this argument is the png output filename where **fx_plot.py** saves the chart.

### Examples of fx_plot.py usage
```bash
$ python fx_plot.py --ds mytestds.csv --prediction myprediction.csv

$ python fx_plot.py --ds mytrainds.csv --prediction myprediction.csv --savefig mychart.png
```


## fx_diag.py<a name="fx_diag"/>
To get the usage of [fx_diag.py](./fx_diag.py) please run
```bash
$ python fx_diag.py --help
```

and you get:
```
usage: fx_diag.py [-h]
                  --dump DUMP_PATH
                  [--savefigdir SAVE_FIGURE_DIRECTORY]

fx_diag.py shows the loss and metric graphs with data generated by fx_fit.py with argument --dumpout

optional arguments:
  -h, --help              show this help message and exit
  --dump        DUMP_PATH dump directory (generated by fx_fit.py with argument --dumpout)
  --savefigdir SAVE_FIGURE_DIRECTORY if present, the charts are saved on different png files in savefig_dir folder instead to be shown on screen
```
Namely:
- **-h or --help** shows the above usage
- **--dump** is the dump directory generated by fx_fit.py with argument --dumpout. This argument is mandatory.
- **--savefigdir** if this argument is missing, the charts are shown on screen one by one, otherwise this argument is the directory where this program saves the graphs in png files; filenames are self-explanatory.

### Examples of fx_diag.py usage
```bash
$ python fx_diag.py --dump dumps/example1

$ python fx_diag.py --dump dumps/example1 --dumps graphs/example1
```


## fx_video.py<a name="fx_video"/>
To get the usage of [fx_video.py](./fx_video.py) please run
```bash
$ python fx_video.py --help
```

and you get:
```
usage: fx_video.py [-h]
                   --modelsnap MODEL_SNAPSHOTS_PATH
                   --ds DATASET_FILENAME
                   --savevideo SAVE_GIF_VIDEO
                   [--fps FPS]
                   [--width WIDTH]
                   [--height HEIGHT]

fx_video.py generates an animated git that shows the prediction curve computed on an input dataset as the epochs change

optional arguments:
  -h, --help              show this help message and exit
  --modelsnap MODEL_SNAPSHOTS_PATH model snapshots directory (generated by fx_fit.py with option --modelsnapout)
  --ds DATASET_FILENAME            dataset file (csv format)
  --savevideo SAVE_GIF_VIDEO       the animated .gif file name to generate
  --fps FPS                        frame per seconds
  --width WIDTH                    width of animated git (in inch)
  --height HEIGHT                  height of animated git (in inch)
```
Namely:
- **-h or --help** shows the above usage
- **--modelsnap** is the directory generated by fx_fit.py with argument --modelsnapout. This argument is mandatory. From this folder this program takes the model snapshots at various times of the epochs and apply the x-values of input dataset to them in order to compute the prediction at the time of the epochs. This argument is mandatory.
- **--ds** input dataset applied to model snapshots (coming from *--modelsnap** parameter) in order to compute the prediction at the time of the epochs. This argument is mandatory.
- **--savevideo** is the animated gif file name to generate. Each frame of this video corresponds to an epoch and it is a figure that shows two overlapped x/y scatter graphs: the blue one is the input dataset, the red one is the prediction one at the time of the epoch. This argument is mandatory.
- **--fps** is the number of frames per second in the final video. This argument is optional.
- **--width** is the animated gif width in inch. This argument is optional.
- **--height** is the animated gif height in inch. This argument is optional.

### Examples of fx_video.py usage
```bash
$ fx_video.py --modelsnap snaps/example1 --ds datasets/example1_test.csv --savevideo graphs/example1_test.gif
```
