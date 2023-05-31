import json
import logging
import os
import string
from collections import Counter

import torch
from PIL import Image

# from lib.utils.caption import Caption
from lib.data.caption import Caption
from lib.data.datasets.dataset_utils import add_special_tokens


from nltk.tokenize import word_tokenize


class FlickrDataset(torch.utils.data.Dataset):
    """
    Dataset loader for Flickr30k and Flickr8k full datasets.
    """

    def __init__(self,
                 root,
                 ann_file,
                 split,
                 transforms=None,
                 cap_transforms=None,
                 use_onehot=True,
                 use_seg=True,
                 use_att=True,
                 max_length=100,
                 max_attribute_length=25,
                 crop_transforms=None,
                 count_thr=2,
                 vocab_path='',
                 debug=False):
        self.logger = logging.getLogger("PersonSearch.lib.data.datasets.flickr")
        self.root = root
        self.transforms = transforms
        self.cap_transforms = cap_transforms
        self.use_onehot = use_onehot
        self.use_seg = use_seg
        self.use_att = use_att
        self.max_length = max_length
        self.max_attribute_length = max_attribute_length
        self.crop_transforms = crop_transforms
        self.vocab_path = vocab_path
        self.split = split

        print(f"loading annotations {ann_file} into memory...")
        dataset = json.load(open(ann_file, "r"))["images"]
        self.dataset = dataset

        self.ids = []
        for i, d in enumerate(self.dataset):
            if d['split'] == split:
                self.ids += [(i, x) for x in range(len(d['sentences']))]

        self.img_ids = [d['imgid'] for d in self.dataset if d['split'] == split]
        img_ids_set = set(self.img_ids)
        self.img_ids_mapper = dict()
        if 'train' in split:
            for idx, img_id in enumerate(img_ids_set):
                self.img_ids_mapper.update({img_id: idx})

        self.ids = self.preprocess_data(self.ids, root, max_length, count_thr)
        if debug and 'train' in split:
            self.ids = self.ids[:1000]

    def __getitem__(self, index):
        root = self.root
        ann_id = self.ids[index]
        img_id = ann_id[0]
        if self.img_ids_mapper:
            label_id = self.img_ids_mapper[img_id]
        else:
            label_id = img_id
        caption = self.dataset[img_id]['sentences'][ann_id[1]]['raw']
        onehot = ann_id[2]
        self.use_onehot=False
        if self.use_onehot:
            caption = torch.tensor(onehot)
            caption = Caption([caption], max_length=self.max_length, padded=False, vocab_path=self.vocab_path)
        else:
            caption = Caption(caption)

        if self.cap_transforms is not None:
            caption = self.cap_transforms(caption)
        label = torch.tensor(label_id)
        caption.add_field("id", label)

        path = self.dataset[img_id]['filename']
        img = Image.open(os.path.join(root, path)).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)

        return img, caption, index

    def __len__(self):
        return len(self.ids)

    def get_id_info(self, index):
        ann_id = self.ids[index]
        image_id = ann_id[0]
        pid = image_id
        return image_id, pid

    def preprocess_data(self, dataset, datadir, max_length, min_word_count=2):
        """
        Append . in the end, and filter out word lower than min_word_count.
        args:
        min_word_count: int.
        """
        outdir = os.path.join(datadir, '..', 'annotations')
        vocab_path = os.path.join(outdir, 'vocab_flickr30k.txt')

        os.makedirs(outdir, exist_ok=True)

        new_dataset = []

        # build vocab
        if not os.path.exists(vocab_path):
            self.logger.info('generate vocab.txt')
            counter = Counter()
            captions = []
            for img_id, cap_id in dataset:
                cap = self.dataset[img_id]['sentences'][cap_id]['raw']
                captions.append(cap)
                counter.update(word_tokenize(cap.lower()))

            # Filter uncommon words and sort by descending count.
            word_counts = [x for x in counter.items() if x[1] > min_word_count]
            word_counts.sort(key=lambda x: x[1], reverse=True)
            self.logger.info("Words in vocabulary: {}".format(len(word_counts)))
            self.logger.info("most words and their counts:")
            self.logger.info("\n".join(map(str, word_counts[:20])))
            self.logger.info("least words and their counts:")
            self.logger.info("\n".join(map(str, word_counts[-20:])))

            # print some stats
            total_words = sum(counter.values())
            self.logger.info(f"total words: {total_words}")
            bad_words = [w for w, n in counter.items() if n <= min_word_count]
            vocab = [w for w, n in counter.items() if n > min_word_count]
            bad_count = sum(counter[w] for w in bad_words)
            self.logger.info(
                "number of bad words: %d/%d = %.2f%%"
                % (len(bad_words), len(counter), len(bad_words) * 100.0 / len(counter))
            )
            self.logger.info("number of words in vocab would be %d" % (len(vocab),))
            self.logger.info(
                "number of UNKs: %d/%d = %.2f%%"
                % (bad_count, total_words, bad_count * 100.0 / total_words)
            )
            # lets look at the distribution of lengths as well
            sent_lengths = {}
            for img_id, cap_id in dataset:
                caption = self.dataset[img_id]['sentences'][cap_id]['raw']
                txt = caption.lower().translate(
                        str.maketrans("", "", string.punctuation)
                    ).strip().split()
                nw = len(txt)
                sent_lengths[nw] = sent_lengths.get(nw, 0) + 1
            max_len = max(sent_lengths.keys())
            self.logger.info(f"max length sentence in raw data: {max_len}")
            self.logger.info("sentence length distribution (count, number of words):")
            sum_len = sum(sent_lengths.values())
            for i in range(max_len + 1):
                self.logger.info(
                    "%2d: %10d   %f%%"
                    % (i, sent_lengths.get(i, 0), sent_lengths.get(i, 0) * 100.0 / sum_len)
                )

            # lets now produce the final annotations
            if bad_count > 0:
                # additional special UNK token we will use below to map infrequent words to
                self.logger.info("inserting the special [UNK] token")
                vocab.append("[UNK]")

            vocab = add_special_tokens(vocab)
            wtoi = {w: i + 1 for i, w in enumerate(vocab)}  # inverse table
            # save vocab
            with open(vocab_path, 'w') as f:
                for word in vocab:
                    f.write(word + '\n')
        else:
            self.logger.info('vocab.txt has been generated.')
            vocab = []
            with open(vocab_path, 'r') as f:
                for word in f.readlines():
                    vocab.append(word.strip())
            wtoi = {w: i + 1 for i, w in enumerate(vocab)}  # inverse table

        for img_id, cap_id in dataset:
            caption = self.dataset[img_id]['sentences'][cap_id]['raw']
            img_path = self.dataset[img_id]['filename']
            onehot = []
            for k, w in enumerate(caption.lower().translate(str.maketrans("", "", string.punctuation)).strip().split()):
                if k < max_length:
                    if w in vocab:
                        onehot.append(wtoi[w])
                    else:
                        assert '[UNK]' in wtoi
                        onehot.append(wtoi['[UNK]'])
            if len(onehot) > 0:
                new_dataset.append((img_id, cap_id, onehot))
            else:
                self.logger.info(f"Wrong caption {caption} for {img_path} found in split {self.split}.")
        return new_dataset