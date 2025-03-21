#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import random

import numpy
import torch


def args_parser():
    parser = argparse.ArgumentParser()
    if torch.cuda.is_available():
        parser.add_argument('--device', type=str, default="cuda:0", help="device")
    else:
        parser.add_argument('--device', type=str, default="cpu", help="device")
    parser.add_argument('--num_workers', type=int, default=1, help="num_workers")
    parser.add_argument('--save_every', type=int, default=1, help="save every xx epoch")
    parser.add_argument('--verbose', type=int, default=1, help='verbose')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--iid', type=int, default=0, help='default non-iid split')
    parser.add_argument('--alg', type=str, required=True, choices=['fedavg', 'fedprox', 'moon', 'scaffold', 'Elastic', 'Hyper', 'fedtp'])
    parser.add_argument('--dataset', type=str, required=True, choices=['smd', 'smap', 'psm', 'swat', 'skab', 'msds','msl','wadi'])
    parser.add_argument('--tsadalg', type=str, required=True, choices=['transformer', 'SCNorTransformer'])
    parser.add_argument('--num_clients', type=int, default=28)
    parser.add_argument('--slide_win', type=int, default=5)
    parser.add_argument('--Loss', type=str, default='time', choices=['fre','time'])
    parser.add_argument('--client_rate', type=float, default=0.7)
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--mu', type=float, default=0.95)
    parser.add_argument('--tau', type=float, default=0.5)
    return parser.parse_args()


args = args_parser()

if args.tsadalg == 'deep_svdd':
    args.slide_win = 1
elif args.tsadalg == 'gdn':
    args.slide_win = 5
elif args.tsadalg == 'lstm_ae':
    args.slide_win = 30
elif args.tsadalg == 'tran_ad':
    args.slide_win = 10
elif args.tsadalg == 'usad':
    args.slide_win = 12
elif args.tsadalg == 'transformer':
    args.slide_win = 20
elif args.tsadalg == 'itransformer':
    args.slide_win = 5

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)


g = torch.Generator()
g.manual_seed(args.seed)

features_dict = {'smd': 38,
                 'smap': 25,
                 'psm': 25,
                 'swat': 51,
                 'skab': 1,
                 'msds': 10,
                 'wadi': 127,
                 'msl': 55
                 }

series_dict = {'smd': 28,
               'smap': 55,
               'psm': 1,
               'swat': 1,
               'skab': 1,
               'msds': 1,
               'wadi': 1,
               'msl': 1
               }

gdn_topk_dict = {'smd': 30,
                 'smap': 20,
                 'psm': 20,
                 'swat': 20,
                 'skab': 20,
                 'msds': 10,
                 'msl': 10
                 }