import torch

from algorithms.Transformer.Transformer import Transformer
from algorithms.Transformer.SCNorTransformer import SCNorTransformer
from datasets.msds_moon import test_dataset, client_datasets_non_iid, client_datasets_iid
from logger import logger
from options import args, features_dict, gdn_topk_dict

config = {
    "epochs": 20,
    "frac": 1,
    "local_ep": 1,
    # "local_bs": 64,
    "local_bs": 128,
    "prox_mu": 0.01,
    "tau": 0.5,
    "moon_mu": 1,
    "optimizer_fun": lambda parameters: torch.optim.Adam(parameters, lr=0.001)
}


if args.tsadalg == 'gdn':
    config["optimizer_fun"] = lambda parameters: torch.optim.Adam(parameters, lr=0.001, weight_decay=0, betas=(0.9, 0.99))
if args.tsadalg == 'deep_svdd' or args.tsadalg == 'deepsvdd':
    config["optimizer_fun"] = lambda parameters: torch.optim.Adam(parameters, lr=0.001, weight_decay=1e-6, betas=(0.9, 0.99))
if args.tsadalg == 'usad':
    config["optimizer_fun"] = lambda parameters: torch.optim.Adam(parameters, lr=0.001, weight_decay=0, betas=(0.9, 0.999))
if args.tsadalg == 'tran_ad':
    if args.dataset == 'smd':
        config["optimizer_fun"] = lambda parameters: torch.optim.AdamW(parameters, lr=0.0001, weight_decay=1e-5)
    elif args.dataset == 'smap':
        config["optimizer_fun"] = lambda parameters: torch.optim.AdamW(parameters, lr=0.001, weight_decay=1e-5)
    elif args.dataset == 'psm':
        config["optimizer_fun"] = lambda parameters: torch.optim.AdamW(parameters, lr=0.001, weight_decay=1e-5)
if args.tsadalg == 'transformer':
    config["optimizer_fun"] = lambda parameters: torch.optim.AdamW(parameters, lr=0.001, weight_decay=1e-5)
if args.tsadalg == 'SCNorTransformer':
    config["optimizer_fun"] = lambda parameters: torch.optim.AdamW(parameters, lr=0.001, weight_decay=1e-5)

if args.iid:
    client_datasets = client_datasets_iid
else:
    client_datasets = client_datasets_non_iid

if args.tsadalg == 'gdn':
    feature_map = get_feature_map(args.dataset)
    fc_struc = get_fc_graph_struc(args.dataset)
    fc_edge_index = build_loc_net(fc_struc, feature_map, feature_map=feature_map)
    fc_edge_index = torch.tensor(fc_edge_index, dtype=torch.long)
    edge_index_sets = []
    edge_index_sets.append(fc_edge_index)
    model_fun = lambda: GDN(edge_index_sets, len(feature_map),
                            dim=features_dict[args.dataset],
                            input_dim=args.slide_win,
                            out_layer_num=1,
                            out_layer_inter_dim=128,
                            topk=gdn_topk_dict[args.dataset]
                            ).to(device=torch.device('cuda:0'))
elif args.tsadalg == 'deep_svdd':
    model_fun = lambda: SMD_MLP_Autoencoder()
elif args.tsadalg == 'usad':
    model_fun = lambda: UsadModel(args.slide_win * features_dict[args.dataset], 100)
elif args.tsadalg == 'tran_ad':
    model_fun = lambda: TranAD(features_dict['msds'])
elif args.tsadalg == 'lstm_ae':
    model_fun = lambda: LSTMAE(n_features=features_dict[args.dataset], hidden_size=128,
                 n_layers=(1, 1), use_bias=(True, True), dropout=(0, 0), device=args.device)
elif args.tsadalg == 'transformer':
    model_fun = lambda: Transformer(enc_in = features_dict[args.dataset], c_out = features_dict[args.dataset])
elif args.tsadalg == 'SCNorTransformer':
    model_fun = lambda: SCNorTransformer(window=args.slide_win)

logger.init(
    f"./result/smd/{args.alg}/seed_{args.seed}"
)