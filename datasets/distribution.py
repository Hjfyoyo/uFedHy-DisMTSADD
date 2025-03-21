import torch
from torch.distributions import Dirichlet
import matplotlib.pyplot as plt
import numpy as np
from options import args
from general_tools import mean, set_random_seed


# def generate_data_nums(num_client, num_data, beta=0.5):
#     while True:
#         data_num_each_client = Dirichlet(torch.tensor([beta] * num_client)).sample()
#         data_num_each_client = torch.floor(num_data * data_num_each_client)
#         data_num_each_client[-1] = num_data - torch.sum(data_num_each_client[:-1])
#         if not (0 in data_num_each_client):
#             break
#     return data_num_each_client

def generate_data_nums(num_client, num_data, beta=0.5):
    while True:
        data_num_each_client = np.random.dirichlet([beta] * num_client) * num_data
        data_num_each_client = np.floor(data_num_each_client).astype(int)
        data_num_each_client[-1] = num_data - np.sum(data_num_each_client[:-1])
        if not (0 in data_num_each_client):
            break
    return data_num_each_client


if __name__ == '__main__':
    length = 708405
    n_clients = 28
    set_random_seed(args.seed)
    data_num_each_client = generate_data_nums(n_clients, length, beta=args.beta)
    data_num_each_client = data_num_each_client.squeeze()

    data_num_each_client = data_num_each_client.astype(int)
    plt.figure(figsize=(12, 8))
    plt.bar(np.arange(n_clients), data_num_each_client, width=0.65, color='#F09BA0', edgecolor='black')
    plt.xticks(np.arange(n_clients))

    # plt.figure(figsize=(12, 8))
    plt.xlabel("Client ID")
    plt.ylabel("Number of samples")
    plt.title("SMD Distribution on Different Clients")
    #F09BA0
    #7ABBDB
    # 显示图形
    plt.savefig('E:/pythonProject/FedTAD-main/plots/Distribution_plots/'+args.dataset+'_beta='+str(args.beta)+'.pdf', dpi=300)
    plt.show()