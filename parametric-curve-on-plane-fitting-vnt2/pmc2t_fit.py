import argparse
import csv
import os
import tensorflow.keras.optimizers as tfo
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model


def build_optimizer():
    optimizer = None
    if args.optimizer_name == 'SGD':
        optimizer = tfo.SGD(lr=args.learning_rate, decay=args.decay, momentum=args.momentum, nesterov=args.nesterov)
    elif args.optimizer_name == 'Adagrad':
        optimizer = tfo.Adagrad(lr=args.learning_rate, epsilon=args.epsilon, decay=args.decay)
    elif args.optimizer_name == 'RMSprop':
        optimizer = tfo.RMSprop(lr=args.learning_rate, rho=args.rho, epsilon=args.epsilon, decay=args.decay)
    elif args.optimizer_name == 'Adadelta':
        optimizer = tfo.Adadelta(lr=args.learning_rate, rho=args.rho, epsilon=args.epsilon, decay=args.decay)
    elif args.optimizer_name == 'Adam':
        optimizer = tfo.Adam(lr=args.learning_rate, beta_1=args.beta_1, beta_2=args.beta_2, epsilon=args.epsilon, decay=args.decay, amsgrad=args.amsgrad)
    elif args.optimizer_name == 'Adamax':
        optimizer = tfo.Adamax(lr=args.learning_rate, beta_1=args.beta_1, beta_2=args.beta_2, epsilon=args.epsilon, decay=args.decay)
    elif args.optimizer_name == 'Nadam':
        optimizer = tfo.Nadam(lr=args.learning_rate, beta_1=args.beta_1, beta_2=args.beta_2, epsilon=args.epsilon, schedule_decay=args.decay)
    return optimizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='pmc2t_fit.py fits a one-variable function in an interval using a configurable multilayer perceptron network')

    parser.add_argument('--trainds',
                        type=str,
                        dest='train_dataset_filename',
                        required=True,
                        help='train dataset file (csv format))')

    parser.add_argument('--modelout',
                        type=str,
                        dest='model_path',
                        required=True,
                        help='output model path')

    parser.add_argument('--epochs',
                        type=int,
                        dest='epochs',
                        required=False,
                        default=500,
                        help='number of epochs)')

    parser.add_argument('--batch_size',
                        type=int,
                        dest='batch_size',
                        required=False,
                        default=50,
                        help='batch size)')

    parser.add_argument('--learning_rate',
                        type=float,
                        dest='learning_rate',
                        required=False,
                        default=0.01,
                        help='learning rate)')

    parser.add_argument('--hlayers',
                        type=int,
                        dest='hidden_layers',
                        required=False,
                        default=1,
                        help='number of hidden layers')

    parser.add_argument('--hunits',
                        type=int,
                        dest='hidden_units',
                        required=False,
                        default=100,
                        help='number of neuors in each hidden layers')

    parser.add_argument('--hactivation',
                        type=str,
                        dest='hidden_activation',
                        required=False,
                        default='relu',
                        help='activation function in hidden layers')

    parser.add_argument('--optimizer',
                        type=str,
                        dest='optimizer_name',
                        required=False,
                        default='Adam',
                        help='optimizer algorithm name')

    parser.add_argument('--decay',
                        type=float,
                        dest='decay',
                        required=False,
                        default=0,
                        help='decay')

    parser.add_argument('--momentum',
                        type=float,
                        dest='momentum',
                        required=False,
                        default=0.0,
                        help='momentum (used only by SGD optimizer)')

    parser.add_argument('--nesterov',
                        type=bool,
                        dest='nesterov',
                        required=False,
                        default=False,
                        help='nesterov (used only by SGD optimizer)')

    parser.add_argument('--epsilon',
                        type=float,
                        dest='epsilon',
                        required=False,
                        default=None,
                        help='epsilon (ignored by SGD optimizer)')

    parser.add_argument('--rho',
                        type=float,
                        dest='rho',
                        required=False,
                        default=0.9,
                        help='rho (used only by RMSprop and Adadelta optimizers)')

    parser.add_argument('--beta_1',
                        type=float,
                        dest='beta_1',
                        required=False,
                        default=0.9,
                        help='beta_1 (used only by Adam, Adamax and Nadam optimizers)')

    parser.add_argument('--beta_2',
                        type=float,
                        dest='beta_2',
                        required=False,
                        default=0.999,
                        help='beta_2 (used only by Adam, Adamax and Nadam optimizers)')

    parser.add_argument('--amsgrad',
                        type=bool,
                        dest='amsgrad',
                        required=False,
                        default=False,
                        help='amsgrad (used only by Adam optimizer)')

    parser.add_argument('--loss',
                        type=str,
                        dest='loss',
                        required=False,
                        default='mean_squared_error',
                        help='loss function name')

    args = parser.parse_args()

    t_train = []
    x_train = []
    y_train = []
    with open(args.train_dataset_filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            t_train.append(float(row[0]))
            x_train.append(float(row[1]))
            y_train.append(float(row[2]))

    inputs_x = Input(shape=(1,))
    hidden = inputs_x
    for i in range(0, args.hidden_layers):
        hidden = Dense(args.hidden_units, activation=args.hidden_activation)(hidden)
    outputs_x = Dense(1)(hidden)
    model_x = Model(inputs=inputs_x, outputs=outputs_x)

    inputs_y = Input(shape=(1,))
    hidden = inputs_y
    for i in range(0, args.hidden_layers):
        hidden = Dense(args.hidden_units, activation=args.hidden_activation)(hidden)
    outputs_y = Dense(1)(hidden)
    model_y = Model(inputs=inputs_y, outputs=outputs_y)

    the_optimizer_x = build_optimizer()
    if the_optimizer_x is None:
        raise Exception('Unknown optimizer {}'.format(args.optimizer_name))

    the_optimizer_y = build_optimizer()

    model_x.compile(loss=args.loss, optimizer=the_optimizer_x)
    model_y.compile(loss=args.loss, optimizer=the_optimizer_y)
    model_x.summary()
    model_y.summary()

    model_x.fit(t_train, x_train, epochs=args.epochs, batch_size=args.batch_size, verbose=1)
    model_y.fit(t_train, y_train, epochs=args.epochs, batch_size=args.batch_size, verbose=1)

    model_x.save(os.path.join(args.model_path, 'x'))
    model_y.save(os.path.join(args.model_path, 'y'))
