import argparse
import csv
import time
import tensorflow.keras.models as tfm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='fx_predict.py makes a prediction on a test dataset of a one-variable function modeled with a pretrained multilayer perceptron network')

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

    x_test = []
    y_test = []
    with open(args.test_dataset_filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            x_test.append(float(row[0]))
            y_test.append(float(row[1]))

    model = tfm.load_model(args.model_path)

    start_time = time.time()
    y_pred = model.predict(x_test, batch_size=1)
    elapsed_time = time.time() - start_time
    print ("Test time:", time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

    csv_output_file = open(args.predicted_data_filename, 'w')
    with csv_output_file:
        writer = csv.writer(csv_output_file, delimiter=',')
        for i in range(0, len(x_test)):
            writer.writerow([x_test[i], y_pred[i][0]])

    print("#### Terminated {} ####".format(__file__));
