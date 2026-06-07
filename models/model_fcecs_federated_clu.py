import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.clustering import sknopp
from lib.metric import map_labels
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, normalized_mutual_info_score, f1_score, adjusted_rand_score


def make_mlplayers(in_channel, cfg, batch_norm=False, out_layer =None):
    layers = []
    in_channels = in_channel
    layer_num = len(cfg)
    for i, v in enumerate(cfg):
        out_channels = v
        mlp = nn.Linear(in_channels, out_channels)
        if batch_norm:
            layers += [mlp, nn.BatchNorm1d(out_channels, affine=False), nn.ReLU()]
        elif i == (layer_num-2):
            layers += [mlp, nn.ReLU()]
        else:
            layers += [mlp]
        in_channels = out_channels
    if out_layer != None:
        mlp = nn.Linear(in_channels, out_layer)
        layers += [mlp]
    return nn.Sequential(*layers)


class FCECS(nn.Module):
    def __init__(self, train_data=None, args=None):
        super(FCECS, self).__init__()
        self.args = args
        self.train_data = train_data
        self.x = train_data.x.to(args.device)
        self.y = train_data.y.to(args.device)
        self.features_num = self.args.features_num
        self.MLP = make_mlplayers(self.features_num, self.args.cfg)
        self.dropout = self.args.dropout
        self.A = None
        self.sparse = True
        self.adj_lists = None
        self.idx_p_list = None
        self.idx_n_list = None
        self.pre_sample_num = 1000
        self.neighbor_epoch = 0
        self.global_centroids = nn.Linear(self.args.cfg[-1], self.args.global_centroids_num, bias=False).requires_grad_(requires_grad=False)
        self.local_centroids = nn.Linear(self.args.cfg[-1], self.args.local_centroids_num, bias=False).requires_grad_(requires_grad=False)
        for m in self.modules():
            self.weights_init(m)
        self.optimiser = torch.optim.Adam(self.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        self.loss_list = []

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def shared_dropout(self, input1, input2, p=0.2):
        mask = torch.rand_like(input1[1]) > p
        mask = mask.repeat(input1.shape[0], 1)
        output1 = input1 * mask
        output2 = input2 * mask
        return output1, output2

    def forward(self, seq_a, feature_k_hop, adj=None):
        h_a_full = self.MLP(seq_a)
        h_p_full = self.MLP(feature_k_hop)

        feature_combine = feature_k_hop
        if self.training is True:
            seq_a, feature_combine = self.shared_dropout(seq_a, feature_combine, self.dropout)
        if self.training is True:
            feature_combine = F.dropout(feature_combine, 0.2, training=self.training)
        h_a = self.MLP(seq_a)
        h_p = self.MLP(feature_combine)

        return h_a, h_p, h_a_full, h_p_full

    @torch.no_grad()
    def embed2(self, seq_a, feature_k_hop,  adj=None):
        self.MLP.eval()
        feature_combine = feature_k_hop
        h_a = self.MLP(seq_a)
        h_p = self.MLP(feature_combine)
        return h_a.detach(), h_p.detach()

    @torch.no_grad()
    def get_embedding_and_label(self):
        h_a, h_p = self.embed2(
            self.x[:, 0:self.features_num*1],
            self.x[:, self.features_num*1:self.features_num*2], None)
        embs = h_p
        embs = embs / embs.norm(dim=1)[:, None]
        embs = embs.detach().cpu()
        labels_true = self.y.detach().cpu().numpy()
        return embs, labels_true

    def get_neighbor_list(self):
        adj_set = {}
        edge_index_0 = self.train_data.edge_index[0]
        edge_index_1 = self.train_data.edge_index[1]
        neighbor_sample = None
        neighbor_round = self.pre_sample_num
        for node_id in range(self.train_data.num_nodes):
            node_neighbor_list = edge_index_1[edge_index_0 == node_id].numpy()
            if len(node_neighbor_list) == 0:
                node_neighbor_list = np.append(node_neighbor_list, node_id)
            adj_set[node_id] = node_neighbor_list
            random_sample = node_neighbor_list[np.random.randint(len(node_neighbor_list), size=neighbor_round)]
            if neighbor_sample is None:
                neighbor_sample = random_sample
            else:
                neighbor_sample = np.vstack((neighbor_sample, random_sample))
        self.idx_p_list = neighbor_sample.T
        self.adj_lists = adj_set

    def get_negative_neighbor_list(self):
        nodes = np.arange(0, len(self.x))
        neighbor_sample = None
        for node in nodes:
            neighbors = set([node])
            frontier = set([node])
            for i in range(1):
                current = set()
                for outer in frontier:
                    current |= set(self.adj_lists[int(outer)])
                frontier = current - neighbors
                neighbors |= current
            far_nodes = set(nodes) - neighbors
            far_nodes_list = np.array(list(far_nodes))
            negative_neighbor = far_nodes_list[np.random.randint(len(far_nodes_list), size=self.pre_sample_num)]
            if neighbor_sample is None:
                neighbor_sample = negative_neighbor
            else:
                neighbor_sample = np.vstack((neighbor_sample, negative_neighbor))
        self.idx_n_list = neighbor_sample.T

    @torch.no_grad()
    def local_clustering(self, node_embed, device="cpu"):
        Z = node_embed
        centroids = Z[
                np.random.choice(Z.shape[0], self.args.local_centroids_num, replace=False)]
        local_iters = 5
        for it in range(local_iters):
            ab_distance = torch.cdist(Z, centroids, p=2)
            ab_distance = 1 / torch.exp(ab_distance)
            assigns = sknopp(ab_distance)
            choice_cluster = torch.argmax(assigns, dim=1)
            for index in range(self.args.local_centroids_num):
                selected = torch.nonzero(choice_cluster == index).squeeze()
                selected = torch.index_select(Z, 0, selected)
                if selected.shape[0] == 0:
                    selected = Z[torch.randint(len(Z), (1,))]
                centroids[index] = selected.mean(dim=0)
        self.local_centroids.weight.data.copy_(centroids.to(device))

    @torch.no_grad()
    def get_global_centroids_min_index(self, full_embedding, global_centroids):
        distances = torch.linalg.norm(full_embedding[:, np.newaxis] - global_centroids, axis=2)
        sample_min_distance_index = torch.argmin(distances, dim=1)
        return sample_min_distance_index

    @torch.no_grad()
    def fine_tuning_negative_centroids_index(self, sample_min_distance_index, negative_sample):
        equal_indices = np.where(sample_min_distance_index == negative_sample)
        for index in equal_indices[0]:
            resample_index = np.random.randint(low=0, high=self.args.global_centroids_num)
            while resample_index == sample_min_distance_index[index]:
                resample_index = np.random.randint(low=0, high=self.args.global_centroids_num)
            negative_sample[index] = resample_index
        return negative_sample


def train_loop(model):
    model.train()
    if model.idx_p_list is None or model.idx_n_list is None:
        model.get_neighbor_list()
        model.get_negative_neighbor_list()
    model.optimiser.zero_grad()
    h_a, h_p, h_a_full, h_p_full = model(
        model.x[:, 0:model.features_num*1],
        model.x[:, model.features_num*1:model.features_num*2], None)
    idx_p_list = model.idx_p_list
    epoch = model.neighbor_epoch
    model.neighbor_epoch += 1
    pre_sample_num = model.pre_sample_num
    h_p_1 = (h_a[idx_p_list[epoch % pre_sample_num]] +
             h_a[idx_p_list[(epoch + 2) % pre_sample_num]] +
             h_a[idx_p_list[(epoch + 4) % pre_sample_num]] +
             h_a[idx_p_list[(epoch + 6) % pre_sample_num]] +
             h_a[idx_p_list[(epoch + 8) % pre_sample_num]]) / 5
    s_p = F.pairwise_distance(h_a, h_p)
    s_p_1 = F.pairwise_distance(h_a, h_p_1)
    idx_list = []
    for i in range(model.args.num_neg):
        neg_idx = np.random.randint(low=0, high=model.pre_sample_num)
        idx_0 = model.idx_n_list[neg_idx]
        idx_list.append(idx_0)
    s_n_list = []
    for h_n in idx_list:
        s_n = F.pairwise_distance(h_a, h_a[h_n])
        s_n_list.append(s_n)

    margin_label = -1 * torch.ones_like(s_p)
    my_margin = model.args.margin1
    my_margin_2 = my_margin + model.args.margin2
    margin_loss = torch.nn.MarginRankingLoss(margin=my_margin, reduction='none')
    lbl_z = torch.tensor([0.]).to(model.args.device)

    loss_mar = 0
    loss_mar_1 = 0
    mask_margin_N = 0
    for s_n in s_n_list:
        loss_mar += (margin_loss(s_p, s_n, margin_label)).mean()
        loss_mar_1 += (margin_loss(s_p_1, s_n, margin_label)).mean()
        mask_margin_N += torch.max((s_n - s_p.detach() - my_margin_2), lbl_z).sum()
    mask_margin_N = mask_margin_N / model.args.num_neg
    loss = loss_mar * model.args.w_loss1 + loss_mar_1 * model.args.w_loss2 + mask_margin_N * model.args.w_loss3

    mask = (torch.rand_like(h_a_full) > 0.2).to(model.args.device)
    clu_s_p = F.pairwise_distance(h_a_full, h_p_full * mask)
    idx_global_centroid_list = []
    if epoch >= model.args.centroids_freeze_epoch:
        sample_min_distance_index = model.get_global_centroids_min_index(h_a_full.detach().clone(), model.global_centroids.weight.data.detach().clone())
        for i in range(model.args.num_neg):
            idx_1 = np.random.randint(model.args.global_centroids_num, size=model.train_data.num_nodes)
            idx_1 = model.fine_tuning_negative_centroids_index(sample_min_distance_index, idx_1)
            idx_global_centroid_list.append(idx_1)
    clu_s_n_list = []
    C = model.global_centroids.weight.data.detach().clone()
    for h_c_n in idx_global_centroid_list:
        s_n = F.pairwise_distance(h_a_full, C[h_c_n] * mask)
        clu_s_n_list.append(s_n)

    clu_loss_mar = 0
    clu_mask_margin_N = 0
    for s_n in clu_s_n_list:
        clu_loss_mar += (margin_loss(clu_s_p, s_n, margin_label)).mean()
        clu_mask_margin_N += torch.max((s_n - clu_s_p.detach() - my_margin_2), lbl_z).sum()
    clu_mask_margin_N = clu_mask_margin_N / model.args.num_neg
    clu_loss = clu_loss_mar * model.args.w_loss1 + clu_mask_margin_N * model.args.w_loss3

    loss = loss + clu_loss
    loss.backward()
    model.optimiser.step()
    with torch.no_grad():
        model.local_clustering(h_p_full)
    model.loss_list.append(float(loss.detach().cpu().numpy()))


def train_FCECS_clu(model, dataloaders, optimizer, local_epoch, device):
    for i in range(local_epoch):
        train_loop(model)


def eval_FCECS_clu(model, client_id):
    model.eval()
    h_a, h_p = model.embed2(
        model.x[:, 0:model.features_num*1],
        model.x[:, model.features_num*1:model.features_num*2], None)
    embs = h_p
    embs = embs / embs.norm(dim=1)[:, None]
    labels_true = model.y.detach().cpu().numpy()
    labels_predict = KMeans(n_clusters=7, n_init='auto').fit_predict(embs.cpu().detach().numpy())
    labels_predict = map_labels(labels_predict, labels_true)[1][labels_predict]
    acc = accuracy_score(labels_true, labels_predict)
    nmi = normalized_mutual_info_score(labels_true, labels_predict, average_method="arithmetic")
    ari = adjusted_rand_score(labels_true, labels_predict)
    f1_micro = f1_score(labels_true, labels_predict, average="micro")
    f1_macro = f1_score(labels_true, labels_predict, average="macro")
    logging.info({"client_id": client_id, "acc": acc, "nmi": nmi, "ari": ari, "f1_mic": f1_micro, "f1_mac": f1_macro})
    logging.info({"client_id": client_id, "loss_list": model.loss_list})
    return 0, acc
