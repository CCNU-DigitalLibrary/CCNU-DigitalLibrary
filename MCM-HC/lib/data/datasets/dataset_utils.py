import string

from lib.utils.logger import setup_logger


logger = setup_logger("PersonSearch.dataset_utils")


def add_start_end(tokens, start_word="<START>", end_word="<END>"):
    """
    Add start and end words for a caption
    """
    tokens_processed = [start_word]
    tokens_processed.extend(tokens)
    tokens_processed.append(end_word)
    return tokens_processed


def add_special_tokens(vocab):
    """
    Add special tokens [PAD], [UNK], [CLS], [SEP], [MASK]
    """
    special_tokens = '[PAD], [UNK], [CLS], [SEP], [MASK]'.split(', ')
    for token in special_tokens:
        if token not in vocab:
            logger.info('inserting the special {} token'.format(token))
            vocab.append(token)
    return vocab


# preprocess all the caption
def prepro_captions(json_ann):
    logger.info("example processed tokens:")
    for i, anno in enumerate(json_ann):
        anno["processed_tokens"] = []
        for j, s in enumerate(anno["captions"]):
            txt = (
                str(s)
                .lower()
                .translate(str.maketrans("", "", string.punctuation))
                .strip()
                .split()
            )
            anno["processed_tokens"].append(txt)


def build_vocab(count_thr, json_ann):
    # count up the number of words
    counts = {}
    for anno in json_ann:
        for txt in anno["processed_tokens"]:
            for w in txt:
                counts[w] = counts.get(w, 0) + 1
    cw = sorted([(count, w) for w, count in counts.items()], reverse=True)
    logger.info("most words and their counts:")
    logger.info("\n".join(map(str, cw[:20])))
    logger.info("least words and their counts:")
    logger.info("\n".join(map(str, cw[-20:])))

    # print some stats
    total_words = sum(counts.values())
    logger.info(f"total words: {total_words}")
    bad_words = [w for w, n in counts.items() if n <= count_thr]
    vocab = [w for w, n in counts.items() if n > count_thr]
    bad_count = sum(counts[w] for w in bad_words)
    logger.info(
        "number of bad words: %d/%d = %.2f%%"
        % (len(bad_words), len(counts), len(bad_words) * 100.0 / len(counts))
    )
    logger.info("number of words in vocab would be %d" % (len(vocab),))
    logger.info(
        "number of UNKs: %d/%d = %.2f%%"
        % (bad_count, total_words, bad_count * 100.0 / total_words)
    )

    # lets look at the distribution of lengths as well
    sent_lengths = {}
    for anno in json_ann:
        for txt in anno["processed_tokens"]:
            nw = len(txt)
            sent_lengths[nw] = sent_lengths.get(nw, 0) + 1
    max_len = max(sent_lengths.keys())
    logger.info(f"max length sentence in raw data: {max_len}")
    logger.info("sentence length distribution (count, number of words):")
    sum_len = sum(sent_lengths.values())
    for i in range(max_len + 1):
        logger.info(
            "%2d: %10d   %f%%"
            % (i, sent_lengths.get(i, 0), sent_lengths.get(i, 0) * 100.0 / sum_len)
        )

    # lets now produce the final annotations
    if bad_count > 0:
        # additional special UNK token we will use below to map infrequent words to
        logger.info("inserting the special [UNK] token")
        vocab.append("[UNK]")

    for anno in json_ann:
        anno["final_captions"] = []
        for txt in anno["processed_tokens"]:
            caption = [w if counts.get(w, 0) > count_thr else "[UNK]" for w in txt]
            anno["final_captions"].append(caption)

    vocab = add_special_tokens(vocab)

    return vocab


# Parse the attribute json file
def parse_att_json(att_list, dictionary):
    att_dict = {
        "head": [],
        "upperbody": [],
        "lowerbody": [],
        "shoe": [],
        "backpack": [],
    }
    for attribute in att_list:
        key = list(attribute.keys())[0]
        values = attribute[key]
        phrase = []

        for word in values:
            if key == "person" and values == ["person"]:
                continue
            if not word.isalpha():
                words = word.replace("/", " ")
                for word in words.split(" "):
                    word = word.lower().translate(
                        str.maketrans("", "", string.punctuation)
                    )
                    if word in dictionary.keys():
                        phrase.append(dictionary[word])
                continue

            word = word.lower().translate(str.maketrans("", "", string.punctuation))
            if word in dictionary.keys():
                phrase.append(dictionary[word])

        if key == "hair" or key == "hat" or key == "person":
            key = "head"
        if key == "other":
            key = "upperbody"
        att_dict[key] += phrase
    return att_dict