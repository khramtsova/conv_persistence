
from scipy.spatial.distance import cdist
import pandas as pd
import time
import numpy as np

import torch
from os import listdir
import wandb

from networks.wideresnet import WideResNet
import argparse
import gudhi as gd
from gudhi import wasserstein


def calculate_topology_per_basic_block(basic_block):
    basic_block_convs = basic_block.conv1, basic_block.conv2
    bd_0_basic_block, bd_1_basic_block = [], []
    per_layer_persistence = []

    for conv_layer in basic_block_convs:
        n_spatial_filters = conv_layer.weight.shape[1]
        bd0_array = [[] for _ in range(n_spatial_filters)]
        bd1_array = [[] for _ in range(n_spatial_filters)]

        per_filter_persistence = []
        filters = conv_layer.weight.detach().cpu()

        for spatial_indx in range(n_spatial_filters):
            # print("Spatial indx", spatial_indx)
            one_spatial_filter = np.array(filters[:, spatial_indx, :, :])
            one_spatial_filter = np.reshape(one_spatial_filter, (-1, 9))
            dist = cdist(one_spatial_filter, one_spatial_filter)
            rips_complex = gd.RipsComplex(distance_matrix=dist)
            st = rips_complex.create_simplex_tree(max_dimension=2)
            st.compute_persistence()  # homology_coeff_field=2)

            bd_0 = st.persistence_intervals_in_dimension(0)
            bd_1 = st.persistence_intervals_in_dimension(1)
            bd0_array[spatial_indx].append(bd_0)
            bd1_array[spatial_indx].append(bd_1)
            per_filter_persistence.append(wasserstein.wasserstein_distance(bd_1, np.array([[0, 0]]),
                                                                           order=1, internal_p=1))

        bd_0_basic_block.append(bd0_array)
        bd_1_basic_block.append(bd1_array)

        per_layer_persistence.append(np.mean(per_filter_persistence))
        # print("Mean and std", np.mean(per_filter_persistence), np.std(per_filter_persistence))

    return bd_0_basic_block, bd_1_basic_block, sum(per_layer_persistence)


def calculate_distances(args):
    # writer = SummaryWriter(log_dir=args.log_dir + logger_id)
    net = WideResNet(depth=16, num_classes=10, widen_factor=4)

    bd0_array_per_epoch = [[] for _ in range(args.max_epoch)]
    bd1_array_per_epoch = [[] for _ in range(args.max_epoch)]

    best_topo = [100000, 100000, 100000]
    best_epoch = [0, 0, 0]
    # rel = [[] for _ in range(n_spatial_filters)]
    # epochs = [i for i in range(500, 50000, 500)]

    for epoch in range(args.min_epoch, args.max_epoch):  # epochs:
        # 2 basic blocks per block
        t1 = time.time()
        bd0_array = [[] for _ in range(2)]
        bd1_array = [[] for _ in range(2)]
        persistence_big_block = [0 for _ in range(2)]

        if epoch % 10 == 0:
            print("Starting epoch", epoch)

        if args.me_ada:
            checkpoints_dir = args.wnb_logdir + "/" + args.model_type + "/" + \
                              args.experiment_id + "/"
            files = listdir(checkpoints_dir)
            file = [file for file in files if 'ep' + str(epoch) + "." in file][0]
            model_dict = torch.load(checkpoints_dir + file, map_location=torch.device('cpu'))

        else:
            checkpoints_dir = args.wnb_logdir + "/" + args.model_type + "/" +\
                          args.experiment_id + "/checkpoints/"

            files = listdir(checkpoints_dir)
            file = [file for file in files if 'epoch='+str(epoch)+"-" in file][0]
            #file = [file for file in files if 'ep' + str(epoch) + ".pth" in file][0]
            checkpoint = torch.load(checkpoints_dir + file, map_location=torch.device('cpu'))["state_dict"]
            # Remove "net." from the lightning checkpoint
            model_dict = {'.'.join(k.split(".")[1:]): v for k, v in checkpoint.items()}

        net.load_state_dict(model_dict, strict=False) #checkpoint)

        blocks = [name for name in net.children()]
        basic_blocks = blocks[args.block_id].layer
        for i, basic_block in enumerate(basic_blocks):
            t3 = time.time()
            bd_0, bd_1, per_block_persistence = calculate_topology_per_basic_block(basic_block)
            wandb.log({"epoch": epoch,
                       "block" + str(args.block_id) + "_bb" + str(i) + "_persistence": per_block_persistence})
            if best_topo[i] > per_block_persistence:
                wandb.run.summary["best_topo_" + str(i)] = per_block_persistence
                wandb.run.summary["best_epoch_" + str(i)] = epoch
                best_topo[i] = per_block_persistence
                best_epoch[i] = epoch
            bd0_array[i] = bd_0
            bd1_array[i] = bd_1
            persistence_big_block[i] = per_block_persistence
            t4 = time.time()
            # print(t4-t3)
        bd0_array_per_epoch[epoch] = bd0_array
        bd1_array_per_epoch[epoch] = bd1_array

        if best_topo[-1] > np.mean(persistence_big_block):
            wandb.run.summary["best_topo_avrg"] = np.mean(persistence_big_block)
            wandb.run.summary["best_epoch_avrg"] = epoch
            best_topo[-1] = np.mean(persistence_big_block)
        t2 = time.time()
        # print("Time required", t2-t1)

    data = pd.DataFrame()
    filename = args.log_dir+args.model_type+"_"+args.experiment_id+"_"+str(args.block_id)+".json"
    with open(filename, 'w') as f:
        data["bd0_array"] = bd0_array_per_epoch
        data["bd1_array"] = bd1_array_per_epoch
        data.to_json(f)
    return bd0_array, bd1_array


def get_args_parser():
    parser = argparse.ArgumentParser()
    # model
    parser.add_argument('--log_dir', default="./log/", type=str)
# "/scratch/user/uqekhram/RandConv/checkpoints/log_wandb/"
    parser.add_argument('--wnb_logdir', default="./log/", type=str)
    parser.add_argument('--model_type', default="RandConv_CIFAR", type=str)
    parser.add_argument('--experiment_id', default="27z5tltz", type=str)
# "/scratch/user/uqekhram/logs/topo_rand_conv_logs"
    parser.add_argument('--min_epoch', default=0, type=int)
    parser.add_argument('--max_epoch', default=200, type=int)
    parser.add_argument('--block_id', default=1, type=int)
    parser.add_argument('--augmentation_type', default="rand_conv", type=str)
    parser.add_argument('--me_ada', action='store_true', help='only consider level5 corruptions')

    return parser.parse_args()


if __name__ == "__main__":

    args = get_args_parser()
    
    # wandb.login(key="3d58dba7b16c49cacee3e18e00b530edc2a87818")
    wandb.init(project="paper_model_selection",
               entity="name",
               # dir="/",
               name=args.experiment_id + "_"+ str(args.block_id),
               job_type=args.augmentation_type,
               config=args)

    calculate_distances(args)


