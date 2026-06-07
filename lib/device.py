import logging
import sys

from lib.client import Client_GC
from lib.server import Server
from models.model_fcecs_federated_clu import FCECS


def setup_device(splitData, server_test_loader, args):
    idx_clients = {}
    clients = []
    model_name = args.model_name
    for index, dataset in enumerate(splitData):
        idx_clients[index] = dataset
        if model_name == "FCECS":
            dataloaders, num_node_features, num_node_classes, train_size = splitData[dataset]
            train = dataloaders["train"]
            if args.backbone == "clu":
                client_model = FCECS(train[0], args)
            else:
                logging.error("please check your backbone, current backbone:{}".format(args.backbone))
            clients.append(Client_GC(client_model, index, dataset, train_size, dataloaders, "", args))
        else:
            logging.error("cannot find client gnn model: {}".format(model_name))
            sys.exit()

    if model_name == "FCECS":
        if args.backbone == "clu":
            server_model = FCECS(server_test_loader[0], args)
        else:
            logging.error("please check your backbone, current backbone:{}".format(args.backbone))
    else:
        logging.error("cannot find server gnn model: {}".format(model_name))
        sys.exit()
    server = Server(server_model, server_test_loader, args)
    return clients, server, idx_clients
