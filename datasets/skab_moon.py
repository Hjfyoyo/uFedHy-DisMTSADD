import numpy as np

from datasets.MOON_util import partition_data, get_dataset
import os
from options import args


current_path = os.getcwd()
data_dir = current_path + '/data/datasets/skab/raw'
train_path = current_path + '/data/datasets/skab/raw/train'
test_path = current_path + '/data/datasets/skab/raw/test'
test_labels_path = current_path + '/data/datasets/skab/raw/test_label'

def skab_noniid():
    train_ds_locals, test_ds_locals = [None] * args.num_clients, [None] * args.num_clients
    chosen_idxes = [i for i in range(args.num_clients)]
    for i in range(len(chosen_idxes)):
        dataidxs = chosen_idxes[i]
        train_ds_locals[i], test_ds_locals[i] = get_dataset(
            "skab", data_dir, dataidxs
        )
    return train_ds_locals


def skab_iid():
    train_ds_locals, test_ds_locals = [None] * args.num_clients, [None] * args.num_clients
    chosen_idxes = [i for i in range(args.num_clients)]
    for i in range(len(chosen_idxes)):
        dataidxs = chosen_idxes[i]
        train_ds_locals[i], test_ds_locals[i] = get_dataset(
            "skab", data_dir, dataidxs
        )
    return train_ds_locals


_, test_dataset = get_dataset(
    "skab",
    data_dir,
    None,
    32
)

client_datasets_non_iid = skab_noniid()
client_datasets_iid = skab_iid()

if __name__ == '__main__':
    _, test_dataset = get_dataset(
        "skab",
        data_dir,
        None,
        32
    )
    for i, (x, labels) in enumerate(test_dataset):
        print(x.shape, labels.shape)