import logging

# import torch.multiprocessing
import torch.utils.data

from lib.config.paths_catalog import DatasetCatalog
from lib.utils.comm import get_world_size, get_rank
from . import datasets as D
from . import samplers
from .collate_batch import collate_fn
from .data_utils import DataLoaderX
from .transforms import build_crop_transforms, build_transforms

# torch.multiprocessing.set_sharing_strategy('file_system')

def build_dataset(
    cfg, dataset_list, transforms, crop_transforms, dataset_catalog, is_train=True
):


    if not isinstance(dataset_list, (list, tuple)):
        raise RuntimeError(
            "dataset_list should be a list of strings, got {}".format(dataset_list)
        )
    datasets = []
    for dataset_name in dataset_list:



        data = dataset_catalog.get(dataset_name)

        # print("check point", cfg.MODE_PERAIN_ON_DOWNSTREAM, dataset_name, data)

        factory = getattr(D, data["factory"])
        args = data["args"]
        args["transforms"] = transforms

        if data["factory"] == "CUHKPEDESDataset":
            args["use_onehot"] = cfg.DATASETS.USE_ONEHOT
            args["use_seg"] = cfg.DATASETS.USE_SEG
            args["use_att"] = cfg.DATASETS.USE_ATT
            args["crop_transforms"] = crop_transforms
            args["max_length"] = cfg.DATASETS.MAX_LENGTH
            args["max_attribute_length"] = cfg.DATASETS.MAX_ATTR_LENGTH
            args["debug"] = cfg.EXP.DEBUG

        # make dataset from factory
        dataset = factory(**args)
        datasets.append(dataset)

    # for testing, return a list of datasets
    if not is_train:
        return datasets

    # for training, concatenate all datasets into a single one
    dataset = datasets[0]
    if len(datasets) > 1:
        dataset = D.ConcatDataset(datasets)

    return [dataset]


def make_data_sampler(cfg, dataset, images_per_gpu, shuffle, is_train, is_val):
    logger = logging.getLogger('PersonSearch.sampler')

    images_per_pid = cfg.DATALOADER.IMS_PER_ID
    if is_train and not is_val:
        sampler_name = cfg.DATALOADER.EN_SAMPLER
        logger.info("Using training sampler {}".format(sampler_name))
        if sampler_name == "TrainingSampler":
            sampler = samplers.TrainingSampler(len(dataset), shuffle)
        elif sampler_name == "NaiveIdentitySampler":
            sampler = samplers.NaiveIdentitySampler(dataset.dataset, images_per_gpu, images_per_pid)
        elif sampler_name == "BalancedIdentitySampler":
            sampler = samplers.BalancedIdentitySampler(dataset.dataset, images_per_gpu, images_per_pid)
        elif sampler_name == "ImbalancedDatasetSampler":
            sampler = samplers.ImbalancedDatasetSampler(dataset.dataset)
        else:
            raise ValueError("Unknown training sampler: {}".format(sampler_name))
    else:
        sampler = samplers.InferenceSampler(len(dataset))

    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, images_per_gpu, drop_last=(is_train and not is_val))
    return batch_sampler


def make_data_loader(cfg, is_train=True, is_val=False):
    logger = logging.getLogger('PersonSearch.dataloader')
    num_gpus = get_world_size()
    if is_train and not is_val:
        images_per_batch = cfg.SOLVER.IMS_PER_BATCH
        assert (
            images_per_batch % num_gpus == 0
        ), "SOLVER.IMS_PER_BATCH ({}) must be divisible by the number of GPUs ({}) used.".format(
            images_per_batch, num_gpus
        )
        images_per_gpu = images_per_batch // num_gpus
        shuffle = True
    else:
        images_per_batch = cfg.TEST.IMS_PER_BATCH
        assert (
            images_per_batch % num_gpus == 0
        ), "TEST.IMS_PER_BATCH ({}) must be divisible by the number of GPUs ({}) used.".format(
            images_per_batch, num_gpus
        )
        images_per_gpu = images_per_batch // num_gpus
        shuffle = False

    if is_train:
        if is_val:
            dataset_list = cfg.DATASETS.VAL
            logger.info('For val dataset:')
        else:
            dataset_list = cfg.DATASETS.TRAIN
            logger.info('For train dataset:')
    else:
        dataset_list = cfg.DATASETS.TEST
        logger.info('For test dataset:')

    transforms = build_transforms(cfg, is_train and not is_val)
    # FIXME: logger is not initialized when testing
    if not logger.isEnabledFor(logging.INFO):
        print(f'build transforms\n {transforms}')
    logger.info(f'build transforms\n {transforms}')

    if cfg.DATASETS.USE_SEG:
        crop_transforms = build_crop_transforms(cfg)
    else:
        crop_transforms = None

    datasets = build_dataset(
        cfg, dataset_list, transforms, crop_transforms, DatasetCatalog, is_train
    )

    data_loaders = []


    for dataset in datasets:
        batch_sampler = make_data_sampler(cfg, dataset, images_per_gpu, shuffle, is_train, is_val)
        num_workers = cfg.DATALOADER.NUM_WORKERS
        # FIXME: use torch.multiprocessing.set_sharing_strategy('file_system') to solve RuntimeError: Pin memory thread exited unexpectedly
        # while master process will be hang on.
        # data_loader = DataLoaderX(
        #     get_rank(),
        #     dataset=dataset,
        #     batch_sampler=batch_sampler,
        #     num_workers=num_workers,
        #     collate_fn=collate_fn,
        #     # pin_memory=True,
        #     # persistent_workers=True
        # )
        data_loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
            # pin_memory=True,
            # persistent_workers=True
        )
        data_loaders.append(data_loader)

    return data_loaders[0]
    # if is_train and not is_val:
    #     # during training, a single (possibly concatenated) data_loader is returned
    #     assert len(data_loaders) == 1
    #     return data_loaders[0]
    # return data_loaders[0]
