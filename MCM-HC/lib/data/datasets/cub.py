import logging
import os
import string
from collections import Counter

import torch
from PIL import Image

from lib.data.caption import Caption
from lib.data.datasets.dataset_utils import add_special_tokens


from nltk.tokenize import word_tokenize


class CUBDataset(torch.utils.data.Dataset):
    def __init__(self,
                 root,
                 caption_root,
                 ann_file,
                 transforms=None,
                 split='train',
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
        self.logger = logging.getLogger('PersonSearch.lib.data.datasets.cub')
        self.transforms = transforms
        self.cap_transforms = cap_transforms
        self.use_onehot = use_onehot
        self.use_seg = use_seg
        self.use_att = use_att
        self.max_length = max_length
        self.max_attribute_length = max_attribute_length
        self.crop_transforms = crop_transforms
        self.vocab_path = vocab_path
        self.img_dir = root
        self.split = split

        with open(ann_file) as fin:
            _classes = [int(line.strip().split('.')[0]) - 1 for line in fin]
        target_classes = set(list(_classes))
        # map the original target_classes to new_classes
        self.target_classes_map = {}
        if 'train' in split:
            for idx, cls_id in enumerate(sorted(list(target_classes))):
                self.target_classes_map.update({cls_id: idx})

        targets = []
        index_to_class = {}
        index_to_img_indices = {}
        class_to_indices = {}
        class_to_img_indices = {}
        idx = 0
        n_images = 0

        for bird_name in os.listdir(root):
            cls_num = int(bird_name.split('.')[0]) - 1
            if cls_num in target_classes:
                _target = []
                for fname in os.listdir(os.path.join(root, bird_name)):
                    txt_fname = os.path.join(caption_root, bird_name, fname.replace('jpg', 'txt'))
                    with open(txt_fname) as fin:
                        captions = [line.strip() for line in fin]

                    n_images += 1
                    class_to_img_indices.setdefault(cls_num, []).append(n_images)
                    for caption in captions:
                        _target.append((os.path.join(root, bird_name, fname), caption))
                        index_to_class[idx] = cls_num
                        index_to_img_indices[idx] = n_images
                        class_to_indices.setdefault(cls_num, []).append(idx)
                        idx += 1
                targets.extend(_target)

        self.dataset = targets
        self.dataset_classes = target_classes
        self.index_to_class = index_to_class
        self.index_to_img_indices = index_to_img_indices
        self.class_to_indices = class_to_indices
        self.class_to_img_indices = class_to_img_indices
        self.n_images = n_images
        self.dataset = self.preprocess_data(self.dataset, root, max_length, count_thr)
        if debug and 'train' in split:
            self.dataset = self.dataset[:1000]
        self.use_onehot=False

    def __getitem__(self, index):
        img_path, caption, onehot = self.dataset[index]

        img = Image.open(img_path).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)

        if self.use_onehot:
            caption = torch.tensor(onehot)
            caption = Caption([caption], max_length=self.max_length, padded=False, vocab_path=self.vocab_path)
        else:
            caption = Caption(caption)

        if self.cap_transforms is not None:
            caption = self.cap_transforms(caption)

        # map the original img_id to new img_id as the class id
        if self.target_classes_map:
            label = self.target_classes_map[self.index_to_class[index]]
        else:
            label = self.index_to_class[index]
        label = torch.tensor(label)
        caption.add_field("id", label)
        caption.add_field("image_path", img_path)

        return img, caption, index

    def __len__(self):
        return len(self.dataset)

    def get_id_info(self, index):
        image_id = self.index_to_img_indices[index]
        pid = self.index_to_class[index]
        return image_id, pid

    def preprocess_data(self, dataset, datadir, max_length, min_word_count=2):
        """
        Append . in the end, and filter out word lower than min_word_count.
        args:
        min_word_count: int.
        """
        outdir = os.path.join(datadir, '..', 'annotations')
        vocab_path = os.path.join(outdir, 'vocab.txt')

        os.makedirs(outdir, exist_ok=True)

        new_dataset = []

        # build vocab
        if not os.path.exists(vocab_path):
            self.logger.info('generate vocab.txt')
            counter = Counter()
            captions = []
            for img_path, cap in dataset:
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
            for _, caption in dataset:
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

        for img_path, caption in dataset:
            onehot = []
            for k, w in enumerate(caption.lower().translate(str.maketrans("", "", string.punctuation)).strip().split()):
                if k < max_length:
                    if w in vocab:
                        onehot.append(wtoi[w])
                    else:
                        assert '[UNK]' in wtoi
                        onehot.append(wtoi['[UNK]'])
            if len(onehot) > 0:
                new_dataset.append((img_path, caption, onehot))
            else:
                self.logger.info(f"Wrong caption {caption} for {img_path} found in split {self.split}.")
        return new_dataset