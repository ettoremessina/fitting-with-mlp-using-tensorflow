import argparse
import csv
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='fx_scatter.py shows two overlapped x/y scatter graphs: the blue one is the dataset, the red one is the prediction')

    parser.add_argument('--ds',
                        type=str,
                        dest='dataset_filename',
                        required=True,
                        help='dataset file (csv format)')

    parser.add_argument('--prediction',
                        type=str,
                        dest='prediction_data_filename',
                        required=True,
                        help='prediction data file (csv format)')

    parser.add_argument('--title',
                        type=str,
                        dest='figure_title',
                        required=False,
                        default='prediction in red',
                        help='if present, it set the title of chart')

    parser.add_argument('--width',
                        type=float,
                        dest='width',
                        required=False,
                        default=19.20,
                        help='width of animated git (in inch)')

    parser.add_argument('--height',
                        type=float,
                        dest='height',
                        required=False,
                        default=10.80,
                        help='height of animated git (in inch)')

    parser.add_argument('--savefig',
                        type=str,
                        dest='save_figure_filename',
                        required=False,
                        default='',
                        help='if present, the chart is saved on a file instead to be shown on screen')

    args = parser.parse_args()

    #print("#### Started {} {} ####".format(__file__, args));
    print("#### Started fx_scatter ####");

    fig, ax = plt.subplots(figsize=(args.width, args.height))

    with open(args.dataset_filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader, None)
        for row in csv_reader:
            plt.scatter(float(row[0]), float(row[1]), color='blue', s=1, marker='.')

    with open(args.prediction_data_filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader, None)
        for row in csv_reader:
            plt.scatter(float(row[0]), float(row[1]), color='red', s=2, marker='.')

    plt.title(args.figure_title);
    if args.save_figure_filename:
        plt.savefig(args.save_figure_filename)
    else:
        plt.show()

    #print("#### Terminated {} ####".format(__file__));
    print("#### Terminated fx_scatter ####");
