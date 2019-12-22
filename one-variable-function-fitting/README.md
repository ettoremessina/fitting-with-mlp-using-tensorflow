# One variable function fitting
This project implements the fitting of a continuous and limited real-valued function defined in a closed interval of the reals.
The function fitting is implemented using a configurable multilayer perceptron neural network written using TensorFlow & Keras; it requires TensorFlow 2.0.0 library.

It contains four python programs:
 - **fx_gen.py** generates a synthetic dataset file invoking a one-variable real function on an real interval.
 - **fx_fit.py** fits a one-variable function in an interval using a configurable multilayer perceptron neural network.
 - **fx_predict.py** makes a prediction on a test dataset of a one-variable function modeled with a pretrained multilayer perceptron neural network.
 - **fx_plot.py** shows two overlapped x/y scatter graphs: the blue one is the train dataset, the red one is the predicted one.

### Predefined examples of usage of the four command in cascade
In the subfolder **examples** there are nine bash scripts to fit nine different one-variable functions; each script executes the four programs in cascade in order to reach and show the goal.

```bash
$ cd examples
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


## fx_gen.py
To get the usage of [fx_gen.py](./fx_gen.py) please run
```bash
$ python fx_gen.py --help
```

and you get
```
fx_gen.py generates a synthetic dataset file calling a one-variable real
function in an interval

optional arguments:
  -h, --help            show this help message and exit
  --dsout DS_OUTPUT_FILENAME
                        dataset output file (csv format)
  --fx FUNC_X_BODY      f(x) body (lamba format)
  --rbegin RANGE_BEGIN  begin range (default:-5.0))
  --rend RANGE_END      end range (default:+5.0))
  --rstep RANGE_STEP    step range (default: 0.01))
```

where:
- **-h or --help** shows the above usage
- **--rbegin** and **--rend** are the limit of the closed interval of reals of independent variable x.
- **--rstep** is the increment step of independent variable x into interval.
- **--fx** is the function to use to compute the value of dependent variable; it is in lamba body format.
- **--dsout** is the target dataset file name. The content of this file is csv and each line contains a couple of real numbers: the x and the f(x) where x is a value of the interval and f(x) is the value of dependent variable; the dataset is sorted by independent variable x. This option is mandatory.

### Example of fx_gen.py usage
```bash
$ python fx_gen.py --dsout mydataset.csv  --fx "np.exp(np.sin(x))" --rbegin -6.0 --rend 6.0 --rstep 0.05
```


## fx_fit.py
To get the usage of [fx_fit.py](./fx_fit.py) please run
```bash
$ python fx_fit.py --help
```

and you get
```
usage: fx_fit.py [-h] --trainds TRAIN_DATASET_FILENAME --modelout MODEL_PATH
                 [--epochs EPOCHS] [--batch_size BATCH_SIZE]
                 [--learning_rate LEARNING_RATE]
                 [--hlayers HIDDEN_LAYERS_LAYOUT [HIDDEN_LAYERS_LAYOUT ...]]
                 [--hactivations ACTIVATION_FUNCTIONS [ACTIVATION_FUNCTIONS ...]]
                 [--optimizer OPTIMIZER] [--loss LOSS]

fx_fit.py fits a one-variable function in an interval using a configurable
multilayer perceptron network

optional arguments:
  -h, --help            show this help message and exit
  --trainds TRAIN_DATASET_FILENAME
                        train dataset file (csv format))
  --modelout MODEL_PATH
                        output model path
  --epochs EPOCHS       number of epochs)
  --batch_size BATCH_SIZE
                        batch size)
  --learning_rate LEARNING_RATE
                        learning rate)
  --hlayers HIDDEN_LAYERS_LAYOUT [HIDDEN_LAYERS_LAYOUT ...]
                        number of neurons for each hidden layers
  --hactivations ACTIVATION_FUNCTIONS [ACTIVATION_FUNCTIONS ...]
                        activation functions between layers
  --optimizer OPTIMIZER
                        optimizer algorithm object
  --loss LOSS           loss function name
```

where:
- **-h or --help** shows the above usage
- **--trainds** is the input training dataset in csv format: a couple of real number for each line respectively for x and y. In case you haven't a such real world true dataset, for your experiments you can generate it synthetically using **fx_gen.py**. This option is mandatory.
- **--modelout** is a non-existing path where the program saves the trained model.
- **--epochs** is the number of epochs of the training process
- **--batch_size** is the batch size
- **--learning_rate** is the learning rate
- **--hlayers** is an array of integers: the size of the array is the number of hidden layers, each value of the array is the number of neurons in the correspondent layer
- **--hactivations** is an array of activation functions: the size of the array must be equals to number of layers and each item of the array is the activation function to apply to the output of  neurons of the correspondent layer
- **--optimizer** is the name of algorithm used by the training process
- **--loss** is the name of loss function used by the training process

### Example of fx_fix.py usage
```bash
$ python fx_fit.py --trainds mytrainds.csv --modelout mymodel \
  --hlayers 120 160 --hactivations tanh relu \
  --epochs 100 --batch_size 50 \
  --optimizer 'Adam(epsilon=1e-07)' --learning_rate 0.05 \
  --loss 'MeanSquaredError()'
```

