import logging
import sys
import torch

from models.model_fcecs_federated_clu import train_FCECS_clu, eval_FCECS_clu

class Client_GC():
    def __init__(self, model, client_id, client_name, train_size, dataLoader, optimizer, args):
        self.model = model.to(args.device)
        self.id = client_id
        self.name = client_name
        self.train_size = train_size
        self.dataLoader = dataLoader
        self.optimizer = optimizer
        self.args = args
        self.model_name = args.model_name

        self.W = {key: value for key, value in self.model.named_parameters()}
        self.dW = {key: torch.zeros_like(value) for key, value in self.model.named_parameters()}
        self.W_old = {key: value.data.clone() for key, value in self.model.named_parameters()}

        self.gconvNames = None

        self.train_stats = ([0], [0], [0], [0])
        self.weightsNorm = 0.
        self.gradsNorm = 0.
        self.convGradsNorm = 0.
        self.convWeightsNorm = 0.
        self.convDWsNorm = 0.

    def download_from_server(self, args, server):
        self.gconvNames = server.W.keys()
        for k in server.W:
            self.W[k].data = server.W[k].data.clone()

    def local_train(self, local_epoch):
        if self.model_name == "FCECS":
            if self.args.backbone == "clu":
                train_stats = train_FCECS_clu(self.model, self.dataLoader, self.optimizer, local_epoch, self.args.device)
            else:
                logging.error("please check your model backbone. current backbone:{}".format(self.args.backbone))
        else:
            logging.error("cannot find client train algorithm for corresponding model: {}".format(self.model_name))
            sys.exit()

        self.train_stats = train_stats
        self.weightsNorm = torch.norm(flatten(self.W)).item()

        weights_conv = {key: self.W[key] for key in self.gconvNames}
        self.convWeightsNorm = torch.norm(flatten(weights_conv)).item()

        grads = {key: value.grad for key, value in self.W.items()}
        grads.pop("global_centroids.weight")
        grads.pop("local_centroids.weight")
        self.gradsNorm = torch.norm(flatten(grads)).item()

        grads_conv = {key: self.W[key].grad for key in self.gconvNames}
        grads_conv.pop("global_centroids.weight")
        grads_conv.pop("local_centroids.weight")
        self.convGradsNorm = torch.norm(flatten(grads_conv)).item()

    @torch.no_grad()
    def evaluate(self):
        if self.args.model_name == "FCECS":
            self.model.eval()
            loss = acc = 0
            if self.args.backbone == "clu":
                loss, acc = eval_FCECS_clu(self.model, self.id)
            else:
                logging.error("please check your model backbone. current backbone:{}".format(self.args.backbone))
        else:
            print("cannot find evaluation algorithm for model {}".format(self.args.model_name))
            sys.exit()
        return loss, acc

    @torch.no_grad()
    def get_predict_and_label(self):
        if self.args.model_name == "FCECS":
            return self.model.get_predict_and_label()
        else:
            print("cannot find evaluation algorithm for model {}".format(self.args.model_name))
            sys.exit()

    @torch.no_grad()
    def get_embedding_and_label(self):
        if self.args.model_name == "FCECS":
            self.model.eval()
            return self.model.get_embedding_and_label()
        else:
            print("cannot find evaluation algorithm for model {}".format(self.args.model_name))
            sys.exit()


def flatten(w):
    return torch.cat([v.flatten() for v in w.values()])
