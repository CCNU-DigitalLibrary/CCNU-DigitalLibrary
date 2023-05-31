import json
import os
import ast
import logging

import torch
from PIL import Image
import random

from lib.utils.caption import Caption
from .dataset_utils import build_vocab, prepro_captions, parse_att_json

import numpy as np

#
from lib.data.datasets.masking_generator import MaskingGenerator_block, MaskGenerator_simmim, MaskGenerator_simmim_original
#



class CUHKPEDESDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root,
        ann_file,
        use_onehot=True,
        use_seg=True,
        use_att=True,
        max_length=100,
        max_attribute_length=25,
        transforms=None,
        crop_transforms=None,
        cap_transforms=None,
        count_thr=2,
        debug=False,
        use_mask=True
    ):

        logger = logging.getLogger("PersonSearch.datasets.cuhkpedes")
        self.root = root
        self.use_onehot = use_onehot
        self.use_seg = use_seg
        self.use_att = use_att
        self.max_length = max_length
        self.max_attribute_length = max_attribute_length
        self.transforms = transforms
        self.crop_transforms = crop_transforms
        self.cap_transforms = cap_transforms



        self.img_dir = os.path.join(self.root, "imgs")
        self.seg_dir = os.path.join(self.root, "segs")




        self.process_json(root, max_length, count_thr)
        logger.info("loading annotations into memory...")
        dataset = json.load(open(ann_file, "r"))
        
        if debug and 'train' in ann_file:
            for k in dataset.keys():
                dataset[k] =  dataset[k][:1000]
        self.dataset = dataset["annotations"]

    def __getitem__(self, index):
        """
        Args:
              index(int): Index
        Returns:
              tuple: (images, labels, captions)
        """
        data = self.dataset[index]


        # read Image data
        img_path = data["file_path"]
        img = Image.open(os.path.join(self.img_dir, img_path)).convert("RGB")
        if self.use_seg:
            seg = Image.open(
                os.path.join(self.seg_dir, img_path.split(".")[0] + ".png")
            )
        else:
            seg = None


        # read Caption data
        if self.use_onehot:
            caption = data["onehot"]
            caption = torch.tensor(caption)
            caption = Caption([caption], max_length=self.max_length, padded=False)
        else:
            caption = data["sentence"]
            caption = Caption(caption)

        caption.add_field("img_path", img_path)

        if self.use_att:
            attribute = data["att_onehot"]
            if isinstance(attribute, dict):
                attribute_list = [torch.tensor(v) for k, v in attribute.items()]
            else:
                attribute_list = [torch.tensor(v) for v in attribute]
            attribute = Caption(
                attribute_list, max_length=self.max_attribute_length, padded=False
            )
            attribute.add_field("mask", attribute.length > 0)
            attribute.length[attribute.length < 1] = 1
            caption.add_field("attribute", attribute)

        label = data["id"]
        label = torch.tensor(label)
        caption.add_field("id", label)

        # gender = data["gender"]
        # gender = torch.tensor(gender)
        # caption.add_field("gender", gender)

        if self.transforms is not None:
            img = self.transforms(img)
            if self.use_seg:
                seg = self.transforms(seg)

        if self.crop_transforms is not None and self.use_seg:
            crops, mask = self.crop_transforms(img, seg)
            caption.add_field("crops", crops)  # value mask
            caption.add_field("mask", mask)  # existence mask

        if self.cap_transforms is not None:
            caption = self.cap_transforms(caption)
        return img, caption, index
        # return caption, index
    def __len__(self):
        return len(self.dataset)

    def get_id_info(self, index):
        image_id = self.dataset[index]["image_id"]
        pid = self.dataset[index]["id"]
        return image_id, pid

    @staticmethod
    def process_json(datadir, max_length, count_thr=2):
        logger = logging.getLogger("PersonSearch.datasets.cuhkpedes")
        outdir = os.path.join(datadir, 'annotations')
        os.makedirs(outdir, exist_ok=True)
        splits = ["train", "val", "test"]
        json_name = "{}.json"

        if all(os.path.exists(os.path.join(outdir, json_name.format(split))) for split in splits):
            logger.info('splits have been generated.')
            return 
        else:
            logger.info('generate splits json.')

        anno_file = "reid_raw.json"
        att_dir = os.path.join(datadir, "text_attribute_graph")
        json_ann = json.load(open(os.path.join(datadir, anno_file)))

        # tokenization and preprocessing
        prepro_captions(json_ann)
        # create the vocab
        vocab = build_vocab(count_thr, json_ann)
        wtoi = {w: i + 1 for i, w in enumerate(vocab)}  # inverse table
        # save vocab
        with open(os.path.join(outdir, 'vocab.txt'), 'w') as f:
            for word in vocab:
                f.write(word + '\n')

        image_id = 0
        for split in splits:
            logger.info("Starting generate {}.json".format(split))
            ann_dict = {}
            annotations = []
            id_collect = {}
            img_collect = {}
            for anno in json_ann:
                if anno["split"] != split:
                    continue

                n = len(anno["final_captions"])
                assert n > 0, "error: some image has no captions"

                image_id += 1
                for cap_idx, cap in enumerate(anno["final_captions"]):
                    ann = {}
                    ann["image_id"] = image_id
                    ann["id"] = anno["id"] - 1
                    id_collect[ann["id"]] = id_collect.get(ann["id"], 0) + 1
                    ann["file_path"] = anno["file_path"]
                    img_collect[ann["file_path"]] = img_collect.get(ann["file_path"], 0) + 1
                    ann["sentence"] = anno["captions"][cap_idx]
                    ann["onehot"] = []
                    for k, w in enumerate(cap):
                        if k < max_length:
                            ann["onehot"].append(wtoi[w])

                    # Load the parsed attribute and write to the processed files - Jacob
                    att_json_file = anno["file_path"].replace("/", "-")
                    if os.path.exists(att_json_file):
                        with open(os.path.join(att_dir, att_json_file + "-" + str(cap_idx) + ".json"), "r") as f:
                            att_dict = parse_att_json(ast.literal_eval(f.read()), wtoi)
                        ann["att_onehot"] = att_dict
                    annotations.append(ann)

            categories = [k for k, v in id_collect.items()]
            ann_dict["categories"] = categories
            ann_dict["annotations"] = annotations
            logger.info("Num id-persons: {}".format(len(id_collect)))
            logger.info("Num images: {}".format(len(img_collect)))
            logger.info("Num annotations: {}".format(len(annotations)))
            with open(os.path.join(outdir, json_name.format(split)), "w") as outfile:
                outfile.write(json.dumps(ann_dict))

def random_word(self, sentence):
    # sentence is a string
    #
    tokens = sentence.split()
    output_label = []

    for i, token in enumerate(tokens):
        prob = random.random()
        if prob < 0.15:
            prob /= 0.15

            # 80% randomly change token to mask token
            if prob < 0.8:
                tokens[i] = self.vocab.mask_index

            # 10% randomly change token to random token
            elif prob < 0.9:
                tokens[i] = random.randrange(len(self.vocab))

            # 10% randomly change token to current token
            else:
                tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)

            output_label.append(self.vocab.stoi.get(token, self.vocab.unk_index))

        else:
            tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)
            output_label.append(0)

    return tokens, output_label

