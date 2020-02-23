import argparse
import csv
import time
import tensorflow.keras.models as tfm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='fxy_predict.py makes prediction of the values of a two-variables real function modeled with a pretrained multilayer perceptron network')

    parser.add_argument('--model',
                        type=str,
                        dest='model_path',
                        required=True,
                        help='model path')

    parser.add_argument('--ds',
                        type=str,
                        dest='dataset_filename',
                        required=True,
                        help='dataset file (csv format); only x-values are used')

    parser.add_argument('--predictionout',
                        type=str,
                        dest='prediction_data_filename',
                        required=True,
                        help='prediction data file (csv format)')

    args = parser.parse_args()

    print("#### Started {} {} ####".format(__file__, args));

    xy_values = []
    with open(args.dataset_filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            xy_values.append((float(row[0]), float(row[1])))

    model = tfm.load_model(args.model_path)
    start_time = time.time()
    z_pred = model.predict(xy_values, batch_size=1)
    elapsed_time = time.time() - start_time
    print ("Test time:", time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

    csv_output_file = open(args.prediction_data_filename, 'w')
    with csv_output_file:
        writer = csv.writer(csv_output_file, delimiter=',')
        for i in range(0, len(xy_values)):
            writer.writerow([xy_values[i][0], xy_values[i][1], z_pred[i][0]])

    print("#### Terminated {} ####".format(__file__));
