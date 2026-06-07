import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import copy
import logging
import numpy as np
import random
import torch

from lib import dataset
from lib import device
from lib import training
from lib import utils
from pathlib import Path
from tensorboardX import SummaryWriter


def set_log():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    BASIC_FORMAT = "%(asctime)s:%(levelname)s:%(message)s"
    DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)

    chlr = logging.StreamHandler()
    chlr.setFormatter(formatter)
    chlr.setLevel(logging.INFO)

    logger.addHandler(chlr)

    logging.info("current pytorch version is:{}".format(torch.__version__))


def process_fedAVG(args, clients, server, summary_writer):
    logging.info("\nDone setting up Federated devices.")

    logging.info("Running fedAVG ...")
    frame = training.run_fedavg(args, clients, server, args.epochs, args.local_epoch, samp=None, summary_writer=summary_writer)
    if args.repeat is None:
        outfile = os.path.join(output_path, "{}_{}_accuracy_fedavg_GC.csv".format(args.split_algorithm, args.split_num))
    else:
        outfile = os.path.join(output_path, "{}_{}_accuracy_fedavg_GC.csv".format(args.split_algorithm, args.split_num))
    frame.to_csv(outfile)
    logging.info(f"Wrote to file: {outfile}")


if __name__ == '__main__':
    set_log()

    args = utils.get_args(
        model_name="FCECS",
        dataset_class="Planetoid",
        dataset_name="Cora",
        task_key="clustering",
    )
    logging.info("algorithm parameters {}".format(args))

    if torch.cuda.is_available() is not True:
        args.device = "cpu"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    logging.info("Preparing date for {} split algorithm and partition {}-{} ..."
          .format(args.dataset_name, args.split_algorithm, args.split_num))
    splitData, df_stats, server_test_loader = dataset.prepareDataset_for_server_and_client(args)
    logging.info("Done")

    output_path = os.path.join(args.output_base, 'raw')
    Path(output_path).mkdir(parents=True, exist_ok=True)
    logging.info(f"Output Path: {output_path}")

    if args.repeat is None:
        output_f = os.path.join(output_path, "{}_{}_stats_trainData.csv".format(args.split_algorithm, args.split_num))
    else:
        output_f = os.path.join(output_path, "{}_{}_{}_stats_trainData.csv".format(args.repeat, args.split_algorithm,
                                                                                   args.split_num))
    df_stats.to_csv(output_f)
    logging.info(f"Wrote to {output_f}")

    init_clients, init_server, init_idx_clients = device.setup_device(splitData, server_test_loader, args)
    logging.info("\nDone setting up devices.")

    sw_path = os.path.join(args.output_base, 'raw', 'tensorboard',
                           "{}_{}_summary_writer".format(args.split_algorithm, args.split_num))
    summary_writer = SummaryWriter(sw_path)

    process_fedAVG(args, clients=copy.deepcopy(init_clients), server=copy.deepcopy(init_server),
                   summary_writer=summary_writer)
