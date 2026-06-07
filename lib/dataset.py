import logging
import networkx as nx
import pandas as pd
import sys
import torch

from torch_geometric.datasets import Planetoid, Amazon, AttributedGraphDataset, Coauthor
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import from_networkx


def get_stats(df, ds, graphs_train, graphs_val=None, graphs_test=None):
    df.loc[ds, "#graphs_train"] = graphs_train[0]
    df.loc[ds, "train_node_num"] = graphs_train[1]

    if graphs_val:
        df.loc[ds, '#graphs_val'] = graphs_val[0]
        df.loc[ds, "val_node_num"] = graphs_val[1]

    if graphs_test:
        df.loc[ds, '#graphs_test'] = graphs_test[0]
        df.loc[ds, "test_node_num"] = graphs_test[1]

    return df


def load_txt_data(edge_path, feat_path):
    G = nx.Graph()
    edge_list = []
    with open(edge_path, 'r') as f:
        for edge in f:
            edge = edge.strip('\n').split(' ')
            if len(edge) == 1:
                edge = edge[0]
                edge = edge.strip('\n').split('\t')
            edge = [eval(x) for x in edge]
            edge_list.append((edge[0], edge[1]))

    G.add_edges_from(edge_list)
    isolated_node_num = 0
    isolated_node_set = []
    feature_data = pd.read_csv(feat_path, header=None, sep=' ')
    for node in feature_data.values:
        node_id = int(node[0])
        node_label = int(node[-1])
        try:
            G.nodes[node_id]['node_feature'] = torch.tensor(node[1:-1], dtype=torch.float32)
            G.nodes[node_id]['node_label'] = node_label
            G.nodes[node_id]['node_id'] = node_id
            G.nodes[node_id]['x'] = torch.tensor(node[1:-1], dtype=torch.float32)
            G.nodes[node_id]['y'] = node_label
            G.nodes[node_id]['train_mask'] = True
            if node_label != -1:
                G.nodes[node_id]['test_mask'] = True
            else:
                G.nodes[node_id]['test_mask'] = False
        except:
            G.add_node(node_id,
                       node_id=node_id,
                       node_feature=torch.tensor(node[1:-1], dtype=torch.float32),
                       node_label=node_label,
                       x=torch.tensor(node[1:-1], dtype=torch.float32),
                       y=node_label,
                       train_mask=True,
                       test_mask=True)
            isolated_node_set.append(node_id)
            isolated_node_num += 1
            pass

    for i in G.nodes:
        if (len(G.nodes[i]) == 0) or (G.nodes[i]['node_feature'] is None):
            logging.info("node {}, has no feature".format(i))

    logging.info("the isolated node num is {}, node set is {}".format(isolated_node_num, isolated_node_set))

    return G


def prepare_data_for_FCECS(local_graph, datasetName, df):
    if type(local_graph) == nx.Graph:
        pyg_graph = from_networkx(local_graph)
    else:
        pyg_graph = local_graph
    edge_split = RandomLinkSplit(
        num_val=0.,
        num_test=0.,
        is_undirected=True,
        add_negative_train_samples=True,
        split_labels=True
    )
    train_data, val_data, test_data = edge_split(pyg_graph)
    df = get_stats(df, datasetName,
                   (1, len(train_data.x[:])),
                   (1, len(val_data.x[:])),
                   (1, len(test_data.x[:])))
    integration_data = (
        {"train": (train_data, local_graph), "val": (val_data), "test": (test_data)},
        train_data.num_node_features,
        len(train_data.y.unique()),
        len(train_data.x)
    )
    return df, integration_data


def prepareDataset_for_server_and_client(args):
    dataset_name = args.dataset_name
    split_algorithm = args.split_algorithm
    split_num = args.split_num
    model_name = args.model_name
    splitData = {}
    df = pd.DataFrame()
    for i in range(split_num):
        edge_file_name = "./data/{}_split/{}/{}/{}_{}_part{}_edge.txt".format(str.lower(dataset_name), split_algorithm, split_num, model_name, split_num, i)
        node_file_name = "./data/{}_split/{}/{}/{}_{}_part{}_node.txt".format(str.lower(dataset_name), split_algorithm, split_num, model_name, split_num, i)
        networkx_G = load_txt_data(edge_file_name, node_file_name)
        datasetName = "dataset" + str(i)
        if model_name == "FCECS":
            df, integration_data = prepare_data_for_FCECS(networkx_G, datasetName, df)
            splitData[datasetName] = integration_data
        else:
            logging.error("cannot find correspond gnn model: {}".format(model_name))
            sys.exit()

    server_test_loader = {}
    if model_name == "FCECS":
        if dataset_name == "Cora" or dataset_name == "CiteSeer" or dataset_name == "PubMed":
            pyg_dataset = Planetoid(root="./data", name=dataset_name)[0]
        if dataset_name == "Photo" or dataset_name == "Computers":
            pyg_dataset = Amazon(root="./data", name=dataset_name)[0]
        if dataset_name == "BlogCatalog":
            pyg_dataset = AttributedGraphDataset(root="./data", name="BlogCatalog")[0]
            pyg_dataset.x = pyg_dataset.x.to_dense()
        if dataset_name == "Physics" or dataset_name == "CS":
            pyg_dataset = Coauthor(root="./data", name="Physics")[0]
        _, integration_data = prepare_data_for_FCECS(pyg_dataset, "server_data", df)
        server_test_loader = integration_data[0]["train"]

    return splitData, df, server_test_loader
