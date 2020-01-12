import argparse
import csv
import os
import time
import tensorflow.keras.models as tfm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='pmc2t_predict.py makes a prediction on a test dataset of a parametric curve on plan modeled with a couple of pretrained multilayer perceptron networks')

    parser.add_argument('--model',
                        type=str,
                        dest='model_path',
                        required=True,
                        help='model path')

    parser.add_argument('--testds',
                        type=str,
                        dest='test_dataset_filename',
                        required=True,
                        help='test dataset file (csv format)')

    parser.add_argument('--predictedout',
                        type=str,
                        dest='predicted_data_filename',
                        required=True,
                        help='predicted data file (csv format)')

    args = parser.parse_args()

    print("#### Started {} {} ####".format(__file__, args));

    t_test = []
    x_test = []
    y_test = []
    with open(args.test_dataset_filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            t_test.append(float(row[0]))
            x_test.append(float(row[1]))
            y_test.append(float(row[2]))

    model_x = tfm.load_model(os.path.join(args.model_path, 'x'))
    model_y = tfm.load_model(os.path.join(args.model_path, 'y'))

    start_time = time.time()
    x_pred = model_x.predict(t_test, batch_size=1)
    y_pred = model_y.predict(t_test, batch_size=1)
    elapsed_time = time.time() - start_time
    print ("Test time:", time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

    csv_output_file = open(args.predicted_data_filename, 'w')
    with csv_output_file:
        writer = csv.writer(csv_output_file, delimiter=',')
        for i in range(0, len(x_test)):
            writer.writerow([t_test[i], x_pred[i][0], y_pred[i][0]])

    print("#### Terminated {} ####".format(__file__));
