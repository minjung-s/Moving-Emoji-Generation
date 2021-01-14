import argparse

def init_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--save_path",
        type = str,
        default = 'generator_exp.pytorch',
        help = 'path to save model'
    )

    parser.add_argument(
        "--data_path",
        type = str,
        default = 'condition_landmarks.csv',
        help = 'dataset path for training'
    )

    parser.add_argument(
        "--t_stamp",
        type = int,
        default = 50,
        help = 'the number of time stamps that you want to generate'
    )

    parser.add_argument(
        "--batch",
        type = int,
        default = 64,
        help = 'batch_size for training'
    )

    parser.add_argument(
        "--num_epochs",
        type = int,
        default = 2,
        help = 'the number of epochs for training'
    )

    parser.add_argument(
        "--conditions",
        type = str,
        default = 'element_mul',
        help = 'choose the way of embedding conditon vector'
    )

    args = parser.parse_args()

    return args
