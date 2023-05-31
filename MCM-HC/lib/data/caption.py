import os.path

import torch

from lib.utils.comm import all_gather


class Caption(object):
    def __init__(
            self, text, length=None, max_length=None, padded=False, dtype=torch.int64, vocab_path=''
    ):
        device = text.device if isinstance(text, torch.Tensor) else torch.device("cpu")
        if isinstance(text, list):
            text = [torch.as_tensor(line, dtype=dtype, device=device) for line in text]
            if length is None:
                length = torch.stack(
                    [
                        torch.tensor(line.size(0), dtype=torch.int64, device=device)
                        for line in text
                    ]
                )
            if max_length is None:
                max_length = max([line.size(-1) for line in text])
        elif isinstance(text, str):
            if length is None:
                length = len(text.split())
        else:
            text = torch.as_tensor(text, dtype=dtype, device=device)
            if length is None:
                length = torch.tensor(text.size(-1), dtype=torch.int64, device=device)
            if max_length is None:
                max_length = text.size(-1)

        if vocab_path:
            self.vocab = {}
            assert os.path.exists(vocab_path), f"{vocab_path} is not exist."
            count = 0
            with open(vocab_path, 'r') as f:
                for word in f.readlines():
                    self.vocab.update({word.strip(): count})
                    count += 1
        else:
            self.vocab = None

        if not padded and not isinstance(text, str):
            text = self.pad(text, max_length, device, self.vocab)

        self.text = text
        self.length = length
        self.max_length = max_length
        self.padded = True
        self.dtype = dtype
        self.extra_fields = {}

    @staticmethod
    def pad(text, max_length, device, vocab):
        padded = []
        for line in text:
            length = line.size(0)
            if length < max_length:
                if vocab and '[PAD]' in vocab:
                    pad_idx = vocab['[PAD]']
                else:
                    pad_idx = 0
                pad = torch.tensor(
                    [pad_idx] * (max_length - length), dtype=torch.int64, device=device
                )
                padded.append(torch.cat((line, pad)))
            else:
                padded.append(line[:max_length])
        return torch.stack(padded)

    def add_field(self, field, field_data):
        self.extra_fields[field] = field_data

    def get_field(self, field):
        return self.extra_fields[field]

    def has_field(self, field):
        return field in self.extra_fields

    def fields(self):
        return list(self.extra_fields.keys())

    # Tensor-like methods

    def to(self, device):
        cap = Caption(
            self.text,
            self.length,
            self.max_length,
            self.padded,
            self.dtype,
        )
        if not isinstance(self.text, str):
            cap.text = cap.text.to(device)
            cap.length = cap.length.to(device)
        for k, v in self.extra_fields.items():
            if hasattr(v, "to"):
                v = v.to(device)
            cap.add_field(k, v)
        return cap

    def __getitem__(self, item):
        cap = Caption(self.text[item], self.max_length, self.padded)
        for k, v in self.extra_fields.items():
            cap.add_field(k, v[item])
        return cap

    def __len__(self):
        return len(self.text)

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "length={}, ".format(self.length)
        s += "max_length={}, ".format(self.max_length)
        s += "padded={}, ".format(self.padded)
        return s


def concate_all_captions(captions):
    texts = [captions[i].text for i in range(len(captions))]
    fields = {}
    for f in captions[0].fields():
        fields.update({f: [captions[i].get_field(f) for i in range(len(captions))]})

    gather_texts = all_gather(texts)
    all_texts = sum(gather_texts, [])
    gather_fields = all_gather(fields)
    all_fields = {}
    for idx, temp_dict in enumerate(gather_fields):
        if idx == 0:
            all_fields.update(temp_dict)
        else:
            for k, v in temp_dict.items():
                all_fields[k].extend(v)
    # construct all_captions
    all_captions = []
    keys = all_fields.keys()
    for i in range(len(all_texts)):
        cap = Caption(all_texts[i])
        for k in keys:
            # if the item is on cuda device, copy these data to the same device
            if isinstance(all_fields[k][i], torch.Tensor) and all_fields[k][i].device.type == 'cuda':
                cap.add_field(k, all_fields[k][i].to('cuda'))
            else:
                cap.add_field(k, all_fields[k][i])

        all_captions.append(cap)
    return all_captions
