import os
import numpy as np
import glob
import config as config
from utils import get_class_mapping
import torch


def prepare_indices_generic():
    C = config.settings['C']
    C_dash = config.settings['C_dash']

    server_root_path = config.server_root_path
    dataset_dir = config.settings['dataset_dir']

    src_datasets = config.settings['src_datasets']
    trgt_datasets = config.settings['trgt_datasets']
    if not os.path.exists(os.path.join('exp', config.settings['exp_name'])):
        os.mkdir(os.path.join('exp', config.settings['exp_name']))

    if not os.path.exists(
            os.path.join('exp', config.settings['exp_name'], config.settings['index_list'])):
        os.mkdir(
            os.path.join('exp', config.settings['exp_name'], config.settings['index_list']))

    shared_trgt_catgs = set([j for x in src_datasets for j in C[x]])
    cat_mapping = get_class_mapping(shared_trgt_catgs, C_dash[src_datasets[0]])

    # using full source dataset for training like MFSAN
    for dataset_name in src_datasets:
        images_train_paths = []
        print('creating index_list for {}'.format(dataset_name))
        for catg in os.listdir(os.path.join(server_root_path, dataset_dir, dataset_name)):
            if catg in C[dataset_name] and catg not in C_dash[dataset_name]:
                cat_id = cat_mapping[catg]
                images_train_paths.extend([[x, cat_id] for x in glob.glob(
                    os.path.join(server_root_path, dataset_dir, dataset_name, catg, '*'))])

        save_path = os.path.join('exp', config.settings['exp_name'],
                                 config.settings['index_list'], '_'.join([dataset_name, 'train.npy']))
        np.save(save_path, images_train_paths)

    # using full target dataset for pseudo labels generation and same target dataset for validation
    for dataset_name in trgt_datasets:

        images_paths = []

        cat_mapping = get_class_mapping(shared_trgt_catgs, C_dash[dataset_name])
        print('creating index_list for target {}'.format(dataset_name))

        for catg in os.listdir(os.path.join(server_root_path, dataset_dir, dataset_name)):
            if catg in shared_trgt_catgs or catg in C_dash[dataset_name]:
                cat_id = cat_mapping[catg]
                images_paths.extend([[x, cat_id] for x in glob.glob(
                    os.path.join(server_root_path, dataset_dir, dataset_name, catg, '*'))])

        save_path = os.path.join('exp', config.settings['exp_name'],
                                 config.settings['index_list'], '_'.join([dataset_name, 'train.npy']))
        np.save(save_path, images_paths)
        save_path = os.path.join('exp', config.settings['exp_name'],
                                 config.settings['index_list'], '_'.join([dataset_name, 'test.npy']))
        np.save(save_path, images_paths)


def prepare_indices_domain_net():
    C = config.settings['C']
    C_dash = config.settings['C_dash']

    server_root_path = config.settings['server_root_path']
    dataset_dir = config.settings['dataset_dir']

    src_datasets = config.settings['src_datasets']
    trgt_datasets = config.settings['trgt_datasets']

    if not os.path.exists(os.path.join('exp', config.settings['exp_name'])):
        os.mkdir(os.path.join('exp', config.settings['exp_name']))

    if not os.path.exists(os.path.join('exp', config.settings['exp_name'], 'index_list')):
        os.mkdir(os.path.join('exp', config.settings['exp_name'], 'index_list'))

    for dataset_name in src_datasets:
        cat_mapping = get_class_mapping(C[dataset_name], C_dash[dataset_name])

        images_path_train = []

        with open(os.path.join(server_root_path, dataset_dir, 'index_main', '_'.join([dataset_name, 'train.txt'])),
                  'r') as f:
            data = f.readlines()
            if config.args.expt == 'paramsen':
                dsize = len(data)
                tr_size = int(0.25*dsize)
                data, _ = torch.utils.data.random_split(data, [tr_size, dsize - tr_size])
            for line in data:
                img_path, class_lbl = line.split(' ')
                _, catg = os.path.split(os.path.split(img_path)[0])
                if catg in C[dataset_name] or catg in C_dash[dataset_name]:
                    cat_id = cat_mapping[catg]
                    images_path_train.append([os.path.join(server_root_path, dataset_dir, img_path.strip()), cat_id])

        save_path = os.path.join('exp', config.settings['exp_name'], 'index_list',
                                 '_'.join([dataset_name, 'train.npy']))
        np.save(save_path, images_path_train)

        print('creating index_list for {}'.format(dataset_name))

    for dataset_name in trgt_datasets:
        images_path_train = []
        images_path_val = []

        shared_trgt_catgs = set([j for x in src_datasets for j in C[x]])

        cat_mapping = get_class_mapping(shared_trgt_catgs, C_dash[dataset_name])

        with open(os.path.join(server_root_path, dataset_dir, 'index_main', '_'.join([dataset_name, 'train.txt'])),
                  'r') as f:
            data = f.readlines()
            if config.args.expt == 'paramsen':
                dsize = len(data)
                tr_size = int(0.25*dsize)
                data, _ = torch.utils.data.random_split(data, [tr_size, dsize - tr_size])
            for line in data:
                img_path, class_lbl = line.split(' ')
                _, catg = os.path.split(os.path.split(img_path)[0])
                if catg in shared_trgt_catgs or catg in C_dash[dataset_name]:
                    cat_id = cat_mapping[catg]
                    images_path_train.append([os.path.join(server_root_path, dataset_dir, img_path.strip()), cat_id])

        with open(os.path.join(server_root_path, dataset_dir, 'index_main', '_'.join([dataset_name, 'test.txt'])),
                  'r') as f:
            data = f.readlines()
            if config.args.expt == 'paramsen':
                dsize = len(data)
                tr_size = int(0.25*dsize)
                data, _ = torch.utils.data.random_split(data, [tr_size, dsize - tr_size])
            for line in data:
                img_path, class_lbl = line.split(' ')
                _, catg = os.path.split(os.path.split(img_path)[0])
                if catg in shared_trgt_catgs or catg in C_dash[dataset_name]:
                    cat_id = cat_mapping[catg]
                    images_path_val.append([os.path.join(server_root_path, dataset_dir, img_path.strip()), cat_id])

        save_path = os.path.join('exp', config.settings['exp_name'], 'index_list',
                                 '_'.join([dataset_name, 'train.npy']))
        np.save(save_path, images_path_train)

        save_path = os.path.join('exp', config.settings['exp_name'], 'index_list',
                                 '_'.join([dataset_name, 'test.npy']))
        np.save(save_path, images_path_val)

        print('creating index_list for {}'.format(dataset_name))


if config.dataset == 'domain-net':
    prepare_indices_domain_net()
else:
    prepare_indices_generic()
