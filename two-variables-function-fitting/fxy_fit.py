import argparse
import csv
import time
import numpy as np
import tensorflow.keras.optimizers as tko
import tensorflow.keras.activations as tka
import tensorflow.keras.losses as tkl
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

def build_model():
    inputs = Input(shape=(2,))
    hidden = inputs
    for i in range(0, len(args.hidden_layers_layout)):
        hidden = Dense(args.hidden_layers_layout[i], activation=build_activation_function(args.activation_functions[i]))(hidden)
    outputs = Dense(1)(hidden)
    model = Model(inputs=inputs, outputs=outputs)
    return model

def build_activation_function(af):
    if af.lower() == 'none':
        return None
    exp_af = 'lambda _ : tka.' + af
    return eval(exp_af)(None)

def build_optimizer():
    opt_init = args.optimizer
    exp_po = 'lambda _ : tko.' + opt_init
    optimizer = eval(exp_po)(None)
    return optimizer

def build_loss():
    exp_loss = 'lambda _ : tkl.' + args.loss
    return eval(exp_loss)(None)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='fxy_fit.py fits a two-variable real function in an interval using a configurable multilayer perceptron network')

    parser.add_argument('--trainds',
                        type=str,
                        dest='train_dataset_filename',
                        required=True,
                        help='train dataset file (csv format)')

    parser.add_argument('--modelout',
                        type=str,
                        dest='model_path',
                        required=True,
                        help='output model directory')

    parser.add_argument('--epochs',
                        type=int,
                        dest='epochs',
                        required=False,
                        default=500,
                        help='number of epochs')

    parser.add_argument('--batch_size',
                        type=int,
                        dest='batch_size',
                        required=False,
                        default=50,
                        help='batch size')

    parser.add_argument('--hlayers',
                        type=int,
                        nargs = '+',
                        dest='hidden_layers_layout',
                        required=False,
                        default=[100],
                        help='number of neurons for each hidden layers')

    parser.add_argument('--hactivations',
                        type=str,
                        nargs = '+',
                        dest='activation_functions',
                        required=False,
                        default=['relu'],
                        help='activation functions between layers')

    parser.add_argument('--optimizer',
                        type=str,
                        dest='optimizer',
                        required=False,
                        default='Adam()',
                        help='optimizer algorithm')

    parser.add_argument('--loss',
                        type=str,
                        dest='loss',
                        required=False,
                        default='MeanSquaredError()',
                        help='loss function name')

    args = parser.parse_args()

    if len(args.hidden_layers_layout) != len(args.activation_functions):
        raise Exception('Number of hidden layers and number of activation functions must be equals')

    print("#### Started {} {} ####".format(__file__, args));

    xy_train = []
    z_train = []
    with open(args.train_dataset_filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            xy_train.append((float(row[0]), float(row[1])))
            z_train.append(float(row[2]))

    model = build_model()

    optimizer = build_optimizer()
    model.compile(loss=build_loss(), optimizer=optimizer)
    model.summary()

    start_time = time.time()
    model.fit(xy_train, z_train, epochs=args.epochs, batch_size=args.batch_size, verbose=1)
    elapsed_time = time.time() - start_time
    print ("Training time:", time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

    model.save(args.model_path)

    print("#### Terminated {} ####".format(__file__));
