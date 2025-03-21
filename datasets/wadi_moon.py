import numpy as np

from datasets.MOON_util import partition_data, get_dataset
import os
from options import args


current_path = os.getcwd()
data_dir = current_path + '/data/datasets/wadi/raw'
train_path = current_path + '/data/datasets/wadi/raw/train'
test_path = current_path + '/data/datasets/wadi/raw/test'
test_labels_path = current_path + '/data/datasets/wadi/raw/test_label'

def wadi_noniid():
    train_ds_locals, test_ds_locals = [None] * args.num_clients, [None] * args.num_clients
    chosen_idxes = [i for i in range(args.num_clients)]
    for i in range(len(chosen_idxes)):
        dataidxs = chosen_idxes[i]
        train_ds_locals[i], test_ds_locals[i] = get_dataset(
            "wadi", data_dir, dataidxs
        )
    return train_ds_locals


def wadi_iid():
    train_ds_locals, test_ds_locals = [None] * args.num_clients, [None] * args.num_clients
    chosen_idxes = [i for i in range(args.num_clients)]
    for i in range(len(chosen_idxes)):
        dataidxs = chosen_idxes[i]
        train_ds_locals[i], test_ds_locals[i] = get_dataset(
            "wadi", data_dir, dataidxs
        )
    return train_ds_locals


_, test_dataset = get_dataset(
    "wadi",
    data_dir,
    None,
    32
)

client_datasets_non_iid = wadi_noniid()
client_datasets_iid = wadi_iid()
