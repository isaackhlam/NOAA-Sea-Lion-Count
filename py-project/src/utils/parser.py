import argparse


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default='cpu', type=str)
    parser.add_argument("--seed", default=42, type=int)


    hyperparam_args = parser.add_argument_group("hyper-parameter")
    hyperparam_args.add_argument("--lr", default=0.001, type=float)
    hyperparam_args.add_argument("--batch_size", default=64, type=int)
    hyperparam_args.add_argument("--epochs", default=20, type=int)
    hyperparam_args.add_argument("--downscale", default=False, type=bool)
    hyperparam_args.add_argument("--patch_size", default=300, type=int)
    hyperparam_args.add_argument("--scale_r", default=0.4, type=float)

    data_args = parser.add_argument_group("data argument")
    data_args.add_argument("--train_data_input_path", default="./data/Train/", type=str)
    data_args.add_argument("--train_data_label_path", default="./data/TrainDotted", type=str)
    data_args.add_argument("--train_misalign_data", default="MismatchedTrainImages.txt", type=str)
    data_args.add_argument("--test_data_path", default="test_X.csv", type=str)
    data_args.add_argument("--data_root_path", default="./data", type=str)
    data_args.add_argument("--shuffle_data", default=True, type=bool)
    data_args.add_argument("--dataloader_num_workers", default=4, type=int)

    return parser

