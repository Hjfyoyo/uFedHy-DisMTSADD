# -*- coding: utf-8 -*-
import os
import sys
import time

import numpy as np
from tqdm import tqdm
from options import args, features_dict
from collections import OrderedDict, defaultdict
from src.diagnosis import *
from typing import List
import torch
from logger import logger
from src.eval_methods import bf_search
from src.plotting import *
from pprint import pprint
from calc_precision_recall_f1_point_adjust import *
from general_tools import mean, set_random_seed
from src.Hypernetworks import ViTHyper #Hyper
# from algorithms.Transformer.Hypernetworks import ViTHyper #FedTP
# from openTSNE import TSNE
# from src.tSNE import utils
from causallearn.search.ConstraintBased.PC import pc
from sknetwork.ranking import PageRank
from causallearn.utils.cit import chisq, fisherz, gsq, kci, mv_fisherz
import networkx as nx

set_random_seed(args.seed)

torch.backends.cudnn.benchmark = False

if args.tsadalg != 'gdn':
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    torch.use_deterministic_algorithms(True)
else:
    torch.backends.cudnn.deterministic = True

from task import config, test_dataset, model_fun, client_datasets, load_model
from clients.client_mix_trans import test_inference, get_init_grad_correct, generate_clients

def average_weights(state_dicts: List[dict], fed_avg_freqs: torch.Tensor):
    # init
    avg_state_dict = {}
    for key in state_dicts[0].keys():
        avg_state_dict[key] = state_dicts[0][key] * fed_avg_freqs[0]

    state_dicts = state_dicts[1:]
    fed_avg_freqs = fed_avg_freqs[1:]
    for state_dict, freq in zip(state_dicts, fed_avg_freqs):
        for key in state_dict.keys():
            avg_state_dict[key] += state_dict[key] * freq
    return avg_state_dict


def update_global_grad_correct(old_correct: dict, grad_correct_deltas: List[dict], fed_avg_freqs: torch.Tensor, num_chosen_client, num_total_client):
    assert (len(grad_correct_deltas) == num_chosen_client)
    total_delta = average_weights(grad_correct_deltas, [1 / num_chosen_client] * num_chosen_client)
    for key in old_correct.keys():
        if key in total_delta.keys():
            old_correct[key] = old_correct[key] + total_delta[key]
    return old_correct

def causallocation(scores):
    cg = pc(scores, 0.05, fisherz, False, 0, -1)
    adj = cg.G.graph
    G = nx.DiGraph()
    for i in range(len(adj)):
        for j in range(len(adj)):
            if adj[i, j] == -1:
                G.add_edge(i, j)
            if adj[i, j] == 1:
                G.add_edge(j, i)
    nodes = sorted(G.nodes())
    adj = np.asarray(nx.to_numpy_array(G, nodelist=nodes))
    pos = nx.circular_layout(G)
    nx.draw(G, pos=pos, with_labels=True)
    plt.savefig(args.alg+'_'+args.dataset+'_Graph.pdf', dpi=600)

    pagerank = PageRank()
    scores = pagerank.fit_transform(adj.T)
    print(scores)
    # cmap = plt.cm.coolwarm

    score_dict = {}
    for i, s in enumerate(scores):
        score_dict[i] = s
    print(sorted(score_dict.items(), key=lambda item: item[1], reverse=True))


def fed_main():
    logger.tik()
    clients = generate_clients(client_datasets)
    print('clients_num:',len(clients))
    model = model_fun().cpu()
    global_state_dict = model.state_dict()
    global_correct = get_init_grad_correct(model_fun().cpu())
    client_nums = len(clients)
    batch_node = int((len(clients) * args.client_rate))
    f1_list = []
    auroc_list = []
    pre_list = []
    rec_list = []
    time_list = []
    if args.alg == 'Hyper':
        hnet = ViTHyper(n_nodes=client_nums, embedding_dim=100, featrue=args.slide_win, hidden_dim=100, dim=128, client_sample=batch_node, depth=3).to(args.device) #Hyper
    else:
        hnet = ViTHyper(n_nodes=client_nums, embedding_dim=100, featrue=features_dict[args.dataset], hidden_dim=100, dim=128, client_sample=batch_node, depth=3).to(args.device)  # FedTP
    # endregion

    optimizers = {
        'sgd': torch.optim.SGD(
            params=hnet.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-3
        ),
        'adam': torch.optim.Adam(params=hnet.parameters(), lr=0.001)
    }
    optimizer = optimizers['sgd']
    #
    # global_para = model_fun.state_dict()

    grads_update = 0
    best_f1_temp = 0
    # best_ap = 0
    print(os.getcwd())
    model_save_path = os.path.abspath(os.getcwd()) + '/fltsad/pths/' + args.alg + '_' + args.tsadalg + '_' + args.dataset + '.pth'
    score_save_path = os.path.abspath(os.getcwd()) + '/fltsad/scores/' + args.alg + '_' + args.tsadalg + '_' + args.dataset + '.npy'
    score_trans_save_path = os.path.abspath(os.getcwd()) + '/fltsad/scores/' + args.alg + '_' + args.tsadalg + '_' + args.dataset +'_trans'+ '.npy'
    label_trans_save_path = os.path.abspath(os.getcwd()) + '/fltsad/scores/' + args.alg + '_' + args.tsadalg + '_' + args.dataset + '_label_trans' + '.npy'

    # Training
    times = []
    for global_round in tqdm(range(config["epochs"]), file=sys.stdout):
        logger.print(f'\n | Global Training Round : {global_round + 1} |\n')


        num_active_client = int((len(clients) * args.client_rate))
        ind_active_clients = np.random.choice(range(len(clients)), num_active_client, replace=False)
        active_clients = [clients[i] for i in ind_active_clients]
        # endregion

        active_state_dict = []
        data_nums = []
        train_accuracies = []
        train_losses = []
        grad_correct_deltas = []
        hnet_grads_list = []
        client_times = []
        node_weights = []
        print('Choose the', ind_active_clients, 'Client')
        hnet.train()
        weights = hnet(torch.tensor([ind_active_clients], dtype=torch.long).cuda(), False)
        idx = 0
        for client in active_clients:
            client_start = time.time()
            data_nums.append(len(client.dataset))
            node_weights.append(weights[idx])
            loss, accuracy, client_final_state = client.local_train(
                global_state_dict,
                global_round,
                weights[idx],
                global_correct,
                )
            client_times.append(time.time() - client_start)
            grad_correct_deltas.append(client_final_state)

            train_losses.append(loss)
            active_state_dict.append(client.state_dict_prev)
            idx = idx + 1

        # end region
        this_time = max(client_times)
        time_start = time.time()
        fed_freq = torch.tensor(data_nums, dtype=torch.float) / sum(data_nums)

        if args.alg == 'Hyper' or args.alg == 'fedtp':
            for idx in range(len(ind_active_clients)):
                net_para = active_state_dict[idx]
                final_state = active_state_dict[idx]
                node_weights = weights[idx]
                inner_state = OrderedDict({k: tensor.data for k, tensor in node_weights.items()})
                delta_theta = OrderedDict({k: inner_state[k] - final_state[k] for k in node_weights.keys()})
                hnet_grads = torch.autograd.grad(
                    list(node_weights.values()), hnet.parameters(), grad_outputs=list(delta_theta.values()),
                    retain_graph=True,
                    allow_unused=True
                )

                if idx == 0:
                    for key in net_para:
                        global_state_dict[key] = net_para[key] * fed_freq[idx]
                    grads_update = [fed_freq[idx] * x for x in hnet_grads]
                else:
                    for key in net_para:
                        global_state_dict[key] += net_para[key] * fed_freq[idx]
                    for g in range(len(hnet_grads)):
                        grads_update[g] += fed_freq[idx] * hnet_grads[g]

            optimizer.zero_grad()
            for p, g in zip(hnet.parameters(), grads_update):
                p.grad = g
            torch.nn.utils.clip_grad_norm_(hnet.parameters(), 50)
        else:
            for idx in range(len(ind_active_clients)):
                net_para = active_state_dict[idx]
                if idx == 0:
                    for key in net_para:
                        global_state_dict[key] = net_para[key] * fed_freq[idx]
                    # grads_update = [fed_freq[idx] * x for x in hnet_grads]
                else:
                    for key in net_para:
                        global_state_dict[key] += net_para[key] * fed_freq[idx]
                    # for g in range(len(hnet_grads)):
                    #     grads_update[g] += fed_freq[idx] * hnet_grads[g]
            optimizer.zero_grad()
            optimizer.step()

        if args.alg == 'scaffold':
            global_correct = update_global_grad_correct(
                global_correct, grad_correct_deltas,
                fed_freq, num_active_client, len(clients)
                )
        # endregion
        time_end = time.time()
        this_time += ((time_end - time_start) / 5)
        times.append(this_time)

        # model = model_fun().cpu()
        # global_state_dict = model.state_dict()
        logger.add_record("train_loss", float(mean(train_losses)), global_round)
        if (global_round + 1) % args.save_every == 0:
            auc_roc, ap, test_loss, scores, labels, scores_trans, labels_trans, pre, true = test_inference(
                load_model(global_state_dict).to(args.device),
                test_dataset
            )
            if args.dataset != 'msds':
                bf_eval = bf_search(scores, labels, start=0.001, end=1, step_num=150,verbose=False)
                pprint(bf_eval)
                f1_list.append(bf_eval['f1'])
                auroc_list.append(bf_eval['AUROC'])
                pre_list.append(bf_eval['precision'])
                rec_list.append(bf_eval['recall'])
                # print('\n')
                # best_auc, best_precision, best_recall, best_f1, best_precision_adjusted, best_recall_adjusted, best_f1_adjusted = get_threshold_2(labels,scores)
                # causallocation(scores_trans)
            else:
                scoresFinal = np.mean(scores, axis=1)
                labelsFinal = (np.sum(labels, axis=1) >= 1) + 0
                bf_eval = bf_search(scoresFinal, labelsFinal, start=0.001, end=1, step_num=150, verbose=False)
                pprint(bf_eval)
                f1_list.append(bf_eval['f1'])
                auroc_list.append(bf_eval['AUROC'])
                pre_list.append(bf_eval['precision'])
                rec_list.append(bf_eval['recall'])
                # print('\n')
                # res = recall_at_k(scores, labels_trans)
                # print(res)
                # print('\n')
                # causallocation(scores)


            logger.add_record("test_auc_roc", auc_roc, global_round + 1)
            logger.add_record("test_ap", ap, global_round + 1)
            logger.add_record("test_loss", test_loss, global_round + 1)
            logger.print(f' \n Results after {global_round + 1} global rounds of training:')
            print('average time:', mean(times))
            time_list.append(mean(times))
            print(f"Test AUC-ROC: {auc_roc}")
            print(f"Test AP: {ap}")

            if bf_eval['f1'] > best_f1_temp:
                best_f1_temp = bf_eval['f1']
                # best_ap = ap
                # plotter(name=(args.alg+'_'+args.dataset), y_true = true, y_pred = pre, ascore = scores_trans, labels = labels_trans)
                # embedding_train = tsne.fit(scores_trans)
                # utils.plot(embedding_train, labels)
                # plt.savefig('E:/pythonProject/FedTAD-main/plots/t-sne/'+args.alg+'_'+args.dataset+'.pdf', dpi=300)
                torch.save(global_state_dict, model_save_path)
                np.save(score_save_path, scores)
                np.save(score_trans_save_path, scores_trans)
                np.save(label_trans_save_path, labels_trans)
            try:
                print(f"Test loss (full sample rate): {test_loss:.2f}")
            except:
                pass
        # endregion

    # print('average time:', mean(times))

    logger.print(f' \n Last Results:')

    print(f"Avg F1: {mean(f1_list)}")
    print(f"Avg AUROC: {mean(auroc_list)}")
    print(f"Avg Precision: {mean(pre_list)}")
    print(f"Avg Recall: {mean(rec_list)}")
    print(f"Avg Time: {mean(time_list)}")

    try:
        print(f"Test loss (full sample rate): {test_loss:.2f}")
    except:
        pass

    logger.tok()
    try:
        logger.save()
    except:
        pass


if __name__ == '__main__':
    fed_main()
