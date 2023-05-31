# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""Centralized catalog of paths."""

import os


class DatasetCatalog:
    DATA_DIR = "datasets"
    DATASETS = {
        # TextReID https://github.com/BrandonHanx/TextReID
        # cuhkpedes
        # train: 34054 images, 68126 captions, 11003 classes
        # val: 3078 images, 6158 captions, 1000 classes
        # test: 3074 images, 6156 captions
        "cuhkpedes_train": {
            "img_dir": "cuhkpedes",
            "ann_file": "cuhkpedes/annotations/train.json",
        },
        "cuhkpedes_val": {
            "img_dir": "cuhkpedes",
            "ann_file": "cuhkpedes/annotations/val.json",
        },
        "cuhkpedes_test": {
            "img_dir": "cuhkpedes",
            "ann_file": "cuhkpedes/annotations/test.json",
        },
        # VSE++ https://github.com/fartashf/vsepp
        # no class id in mscoco, we use image id as class id
        # mscoco
        # trainrestval: 113287 images, 566435 captions, 5 captions per image
        # train: 82783 images, 413915 captions
        # val: 5000 images, 25000 captions
        # test: 5000 images, 25000 captions
        "mscoco_trainrv": {
            "img_dir": ("mscoco/train2014", "mscoco/val2014"),
            "ann_file": ("mscoco/annotations/captions_train2014.json", "mscoco/annotations/captions_val2014.json"),
            "ids": ("mscoco/annotations/coco_train_ids.npy", "mscoco/annotations/coco_restval_ids.npy"),
            "split": "trainrestval"
        },
        "mscoco_train": {
            "img_dir": "mscoco/train2014",
            "ann_file": "mscoco/annotations/captions_train2014.json",
            "ids": "mscoco/annotations/coco_train_ids.npy",
            "split": "train"
        },
        "mscoco_val": {
            "img_dir": "mscoco/val2014",
            "ann_file": "mscoco/annotations/captions_val2014.json",
            "ids": "mscoco/annotations/coco_dev_ids.npy",
            "split": "val"
        },
        "mscoco_test": {
            "img_dir": "mscoco/val2014",
            "ann_file": "mscoco/annotations/captions_val2014.json",
            "ids": "mscoco/annotations/coco_test_ids.npy",
            "split": "test"
        },
        # Flickr30k and Flickr8k, download from kaggle
        # no class id in flickr, we use image id as class id
        # flickr30k
        # train: 29k images, 145k captions
        # val: 1014 images, 5070 captions
        # test: 1k images, 5k captions
        # flickr8k
        # train: 6k images, 30k captions
        # val: 1k images, 5k captions
        # test: 1k images, 5k captions
        "flickr30k_train": {
            "img_dir": "flickr/flickr30k-images",
            "ann_file": "flickr/annotations/dataset_flickr30k.json",
            "split": "train",
        },
        "flickr30k_val": {
            "img_dir": "flickr/flickr30k-images",
            "ann_file": "flickr/annotations/dataset_flickr30k.json",
            "split": "val",
        },
        "flickr30k_test": {
            "img_dir": "flickr/flickr30k-images",
            "ann_file": "flickr/annotations/dataset_flickr30k.json",
            "split": "test",
        },
        # "flickr8k_train": {
        #     "img_dir": "flickr/flickr8k-images",
        #     "ann_file": "flickr/annotations/dataset_flickr8k.json",
        #     "split": "train",
        # },
        # "flickr8k_val": {
        #     "img_dir": "flickr/flickr8k-images",
        #     "ann_file": "flickr/annotations/dataset_flickr8k.json",
        #     "split": "val",
        # },
        # "flickr8k_test": {
        #     "img_dir": "flickr/flickr8k-images",
        #     "ann_file": "flickr/annotations/dataset_flickr8k.json",
        #     "split": "test",
        # },
        # train val split used in https://github.com/reedscot/cvpr2016
        # trainval: 8855 images, 88550 captions, 150 classes
        # train: 5894 images, 58940 captions, 100 classes
        # val: 2961 images, 29610 captions, 50 classes
        # test: 2933 images, 29330 captions, 50 classes
        "cub_train": {
            "img_dir": "cub/images",
            "caption_dir": "cub/text_c10",
            "ann_file": "cub/annotations/trainclasses.txt",
            "split": "train",
        },
        "cub_val": {
            "img_dir": "cub/images",
            "caption_dir": "cub/text_c10",
            "ann_file": "cub/annotations/valclasses.txt",
            "split": "val",
        },
        "cub_trainval": {
            "img_dir": "cub/images",
            "caption_dir": "cub/text_c10",
            "ann_file": "cub/annotations/trainvalclasses.txt",
            "split": "trainval",
        },
        "cub_test": {
            "img_dir": "cub/images",
            "caption_dir": "cub/text_c10",
            "ann_file": "cub/annotations/testclasses.txt",
            "split": "test",
        },
        # Oxford flowers
        # download from https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html
        # trainval: 7034 images, 70340 captions, 82 classes
        # train: 5878 images, 58780 captions, 62 classes
        # val: 1156 images, 11560 captions, 20 classes
        # test: 1155 images, 11550 captions, 20 classes
        "flowers_train": {
            "img_dir": "flowers/jpg",
            "caption_dir": "flowers/text_c10",
            "ann_file": "flowers/annotations/trainclasses.txt",
            "split": "train",
        },
        "flowers_val": {
            "img_dir": "flowers/jpg",
            "caption_dir": "flowers/text_c10",
            "ann_file": "flowers/annotations/valclasses.txt",
            "split": "val",
        },
        "flowers_trainval": {
            "img_dir": "flowers/jpg",
            "caption_dir": "flowers/text_c10",
            "ann_file": "flowers/annotations/trainvalclasses.txt",
            "split": "trainval",
        },
        "flowers_test": {
            "img_dir": "flowers/jpg",
            "caption_dir": "flowers/text_c10",
            "ann_file": "flowers/annotations/testclasses.txt",
            "split": "test",
        },

    }

    @staticmethod
    def get(name):
        if "cuhkpedes" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["img_dir"]),
                ann_file=os.path.join(data_dir, attrs["ann_file"]),
            )
            return dict(
                factory="CUHKPEDESDataset",
                args=args,
            )
        elif "mscoco" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            try:
                args = dict(
                    root=os.path.join(data_dir, attrs["img_dir"]),
                    ann_file=os.path.join(data_dir, attrs["ann_file"]),
                    ids=os.path.join(data_dir, attrs["ids"]),
                    split=attrs["split"]
                )
            except:
                assert 'trainrv' in name, "double ann_file used in trainrv"
                args = dict(
                    root=tuple([os.path.join(data_dir, img_dir) for img_dir in attrs["img_dir"]]),
                    ann_file=tuple([os.path.join(data_dir, ann_file) for ann_file in attrs["ann_file"]]),
                    ids=tuple([os.path.join(data_dir, ids) for ids in attrs["ids"]]),
                    split=attrs["split"]
                )
            return dict(
                factory="COCODataset",
                args=args,
            )
        elif "flickr" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["img_dir"]),
                ann_file=os.path.join(data_dir, attrs["ann_file"]),
                split=attrs["split"]
            )
            return dict(
                factory="FlickrDataset",
                args=args,
            )
        elif "cub" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["img_dir"]),
                caption_root=os.path.join(data_dir, attrs["caption_dir"]),
                ann_file=os.path.join(data_dir, attrs["ann_file"]),
                split=attrs["split"]
            )
            return dict(
                factory="CUBDataset",
                args=args,
            )
        elif "flowers" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["img_dir"]),
                caption_root=os.path.join(data_dir, attrs["caption_dir"]),
                ann_file=os.path.join(data_dir, attrs["ann_file"]),
                split=attrs["split"]
            )
            return dict(
                factory="FlowersDataset",
                args=args,
            )
        raise RuntimeError("Dataset not available: {}".format(name))
