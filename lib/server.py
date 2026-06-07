import random
import sys
import numpy as np
import torch
import torch.nn.functional as F

from lib.clustering import sknopp
from lib.metric import eval_for_multi_client


class Server():
    def __init__(self, model, server_test_loader, args):
        self.model = model.to(args.device)
        self.W = {key: value for key, value in self.model.named_parameters()}
        self.model_cache = []
        self.args = args
        self.dataLoader = server_test_loader

    def randomSample_clients(self, all_clients, frac):
        return random.sample(all_clients, int(len(all_clients) * frac))

    def aggregate_weights(self, selected_clients):
        total_size = 0
        for client in selected_clients:
            total_size += client.train_size
        client_local_centroids_list = None
        for k in self.W.keys():
            if k == "local_centroids.weight":
                for client in selected_clients:
                    local_centroids = client.W[k].data
                    if client_local_centroids_list is None:
                        client_local_centroids_list = local_centroids
                    else:
                        client_local_centroids_list = torch.concat([client_local_centroids_list, local_centroids], dim=0)
                global_centroids = self.global_clustering(client_local_centroids_list)
                self.W["global_centroids.weight"].data = global_centroids
                continue
            if k == "global_centroids.weight":
                continue
            self.W[k].data = torch.div(torch.sum(torch.stack([torch.mul(client.W[k].data, client.train_size) for client in selected_clients]), dim=0), total_size).clone()

    @torch.no_grad()
    def global_clustering(self, node_embed, device="cpu"):
        with torch.no_grad():
            Z = node_embed
            centroids = Z[
                np.random.choice(Z.shape[0], self.args.global_centroids_num, replace=False)]
            local_iters = 5
            for it in range(local_iters):
                ab_distance = torch.cdist(Z, centroids, p=2)
                ab_distance = 1 / torch.exp(ab_distance)
                assigns = sknopp(ab_distance)
                choice_cluster = torch.argmax(assigns, dim=1)
                for index in range(self.args.global_centroids_num):
                    selected = torch.nonzero(choice_cluster == index).squeeze()
                    selected = torch.index_select(Z, 0, selected)
                    if selected.shape[0] == 0:
                        selected = Z[torch.randint(len(Z), (1,))]
                    centroids[index] = selected.mean(dim=0)
            return centroids

    def evaluate_multi_client(self, predictList, trueList):
        if self.args.model_name == "FCECS":
            eval_for_multi_client(predictList, trueList)
        else:
            print("cannot find evaluation algorithm for model {}".format(self.args.model_name))
            sys.exit()
