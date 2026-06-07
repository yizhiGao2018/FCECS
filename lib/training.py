import logging
import pandas as pd
import numpy as np

from lib.metric import calculate_predict


def run_fedavg(args, clients, server, COMMUNICATION_ROUNDS, local_epoch, samp=None, frac=1.0, summary_writer=None):
    for client in clients:
        client.download_from_server(args, server)

    if samp is None:
        sampling_fn = server.randomSample_clients
        frac = 1.0

    for c_round in range(1, COMMUNICATION_ROUNDS + 1):
        if (c_round) % 50 == 0:
            logging.info(f"  > round {c_round}")

        if c_round == 1:
            selected_clients = clients
        else:
            selected_clients = sampling_fn(clients, frac)

        for client in selected_clients:
            client.local_train(local_epoch)

        server.aggregate_weights(selected_clients)
        for client in selected_clients:
            client.download_from_server(args, server)

    frame = pd.DataFrame()
    for client in clients:
        loss, acc = client.evaluate()
        frame.loc[client.name, 'test_acc'] = acc

    def highlight_max(s):
        is_max = s == s.max()
        return ['background-color: yellow' if v else '' for v in is_max]

    global_evaluate(server, clients)

    fs = frame.style.apply(highlight_max).data
    print(fs)
    return frame


def global_evaluate(server, clients):
    index = 0
    embedding_list = []
    label_list = []
    for client in clients:
        embedding, label = client.get_embedding_and_label()
        if index == 0:
            embedding_list = embedding
            label_list = label
            index += 1
        else:
            embedding_list = np.vstack((embedding_list, embedding))
            label_list = np.hstack((label_list, label))
    predict_label, true_label = calculate_predict(embedding_list, label_list)
    server.evaluate_multi_client(predict_label, true_label)
