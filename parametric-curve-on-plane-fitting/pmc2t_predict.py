import argparse
import csv
import time
import tensorflow.keras.models as tfm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='pmc2t_predict.py makes prediction of couples of coordinates of a parametric curve on plan modeled with a pretrained multilayer perceptron with two output neurons')

    parser.add_argument('--model',
                        type=str,
                        dest='model_path',
                        required=True,
                        help='model directory')

    parser.add_argument('--ds',
                        type=str,
                        dest='dataset_filename',
                        required=True,
                        help='dataset file (csv format); only t-values are used')

    parser.add_argument('--predictionout',
                        type=str,
                        dest='prediction_data_filename',
                        required=True,
                        help='prediction data file (csv format)')

    args = parser.parse_args()

    print("#### Started {} {} ####".format(__file__, args));

    t_values = []
    with open(args.dataset_filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            t_values.append(float(row[0]))

    model = tfm.load_model(args.model_path)

    start_time = time.time()
    x_pred, y_pred = model.predict(t_values, batch_size=1)
    elapsed_time = time.time() - start_time
    print ("Prediction time:", time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

    csv_output_file = open(args.prediction_data_filename, 'w')
    with csv_output_file:
        writer = csv.writer(csv_output_file, delimiter=',')
        for i in range(0, len(t_values)):
            writer.writerow([t_values[i], x_pred[i][0], y_pred[i][0]])

    print("#### Terminated {} ####".format(__file__));
