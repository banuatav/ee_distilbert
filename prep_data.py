import re
import os
import shutil
import gdown
from pathlib import Path

from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast
import numpy as np
import torch

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


def read_conll(file_path):
    """
    Reads .conll file which has a new token and its entity on each line. Documents are split by an empty line.

    Parameters:
    file_path (string): Path to .conll file

    Returns:
    token_docs (list): contains lists of strings, each list of each documents of tokens
    tag_docs (list): contains lists of strings, each list of each documents of entity tags

    """

    # read file from path
    file_path = Path(file_path)
    raw_text = file_path.read_text().strip()

    # split documents
    raw_docs = re.split(r'\n\t?\n', raw_text)

    # iterate over docs to extract tokens and tags
    token_docs = []
    tag_docs = []
    for doc in raw_docs:
        tokens = []  # holds a list for each doc of tokens
        tags = []  # holds a list for each doc of tags
        for line in doc.split('\n'):
            token, tag = line.split('\t')
            tokens.append(token)
            tags.append(tag)
        token_docs.append(tokens)
        tag_docs.append(tags)

    return token_docs, tag_docs


def encode_tags(tags, encodings, tag2id):
    """
    Encodes tags to corresponding ids and adjusts offsetting.

    Parameters:
    tags (list): contains a list of tags of every token for each document
    encodings (object 'transformers.tokenization_utils_base.BatchEncoding'): 
                encodings returned by the tokenizer of each document,
                is a dictionary containing keys input_ids, attention_mask, offset_mapping
    tag2id (dic): mapping of each tag to its id

    Returns
    encoded_labels (list): contains a list of tag ids for each document padded with -100 for tokens that arent the first part of the original token.
    """

    # encode tags to ids
    tag_ids = [[tag2id[tag] for tag in doc] for doc in tags]

    # adjust for offsetting
    labels_encoded = []
    for labels_i, offset_i in zip(tag_ids, encodings["offset_mapping"]):
        # create an empty array of -100
        labels_encoded_i = np.ones(len(offset_i), dtype=int) * -100
        array_offset_i = np.array(offset_i)

        # -- Everything that starts with 0 and ends with non-zero number is first part of original token.
        # -- Everything that starts and ends with 0 is a special token like [PAD] or [CLS] or [SEP].
        # set labels whose first offset position is 0 and the second is not 0
        labels_encoded_i[(array_offset_i[:, 0] == 0) & (
            array_offset_i[:, 1] != 0)] = labels_i  # select all indexes with starts with 0 and doesnt end with 0
        labels_encoded.append(labels_encoded_i.tolist())

    return labels_encoded


def understand_offset(train_texts, train_tags, train_encodings, tokenizer):
    """
    Conclusions: 
    - (0,0) is special tokens for start and end sequence
    - (0, not0) is begin token of original token
    - (not0, 0) is follow-up token of original token
    """
    print()
    print("-"*100)

    offset_text = train_encodings["offset_mapping"][0]
    input_ids = train_encodings["input_ids"][0]
    attention_mask = train_encodings["attention_mask"][0]

    text = train_texts[0]
    joined_text = " ".join(text)
    tokenized_text = tokenizer.tokenize(" ".join(text))
    tokenized_text = ["BEGIN_TOKEN"]+tokenized_text + \
        ["END_TOKEN"] + [0 for x in attention_mask if x == 0]

    print("Text: '{}'".format(joined_text))
    print("--")

    count = 0
    for text, offset, input, attention in zip(tokenized_text, offset_text, input_ids, attention_mask):
        if not str(offset) == '(0, 0)' and '(0, ' in str(offset):
            count += 1
        print("Text = {}".format(text))
        print("Offset = {}".format(offset))
        print("Input ID = {}".format(input))
        print("Attention Mask = {}".format(attention))
        print("--")

    print("Number of tags = {} and number of (0,0) offsets = {}".format(
        len(train_tags[0]), count))
    print("-"*100)
    print()


class EEDataset(torch.utils.data.Dataset):
    """
    torch.utils.data.Dataset is an abstract class representing a dataset. Your custom dataset should inherit Dataset and override the following methods:

    __len__ so that len(dataset) returns the size of the dataset.
    __getitem__ to support the indexing such that dataset[i] can be used to get ith sample

    Parameters:
    encodings (dict): containing input_ids and attention_mask for each doc
    labels (list): containing lists of labels for each token
    """

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    # returns a dictionary containing values of input_id, attention_mask and label for index 'idx'
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx])
                for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def create_dataset(file_path, verbose=False):
    """
    Creates training and valuation datasets for training. Steps are:
    1. Extract lists of tags and docs using read_conll()
    2. Perform train-val split using sklearn
    3. Create set of unique tags
    4. Create tag ids + dictionarie tag2id and id2tag
    5. Load tokenizer from pretrained HF Tokenizers, returns offset mapping to be able to adjust tags
    6. Create encodings for tokens: uses tokenizer to create input ids and attentionmask
    7. Adjust tags for offsettting using encode_tags()

    Returns:
    train_dataset (object of EEDataset)
    train_tags (list)
    val_dataset (object of EEDataset)
    val_tags (list)
    unique_tags (set)
    id2tag (dict)
    """

    print("-- Reading data..")
    texts, tags = read_conll(file_path)

    print("-- Train-val split data..")
    train_texts, val_texts, train_tags, val_tags = train_test_split(
        texts, tags, test_size=.2)

    print("-- Creating set of unique tags..")
    unique_tags = set(tag for doc in tags for tag in doc)

    print("-- Creating dictionary of unique tag 2 ids..")
    tag2id = {tag: id for id, tag in enumerate(unique_tags)}

    print("-- Creating dictionary of unique ids 2 tags..")
    id2tag = {id: tag for tag, id in tag2id.items()}

    print("-- Loading tokenizer..")
    tokenizer = DistilBertTokenizerFast.from_pretrained(
        'distilbert-base-cased')

    print("-- Creating encodings..")
    train_encodings = tokenizer(train_texts, is_pretokenized=True,
                                return_offsets_mapping=True, padding=True, truncation=True)
    val_encodings = tokenizer(val_texts, is_pretokenized=True,
                              return_offsets_mapping=True, padding=True, truncation=True)

    if verbose:
        understand_offset(train_texts, train_tags, train_encodings, tokenizer)

    print("-- Adjusting labels for offsetting ..")
    train_tag_ids_offadj = encode_tags(
        train_tags, train_encodings, tag2id)
    val_tag_ids_offadj = encode_tags(val_tags, val_encodings, tag2id)

    # remove offset mapping now it is no longer needed
    train_encodings.pop("offset_mapping")
    val_encodings.pop("offset_mapping")

    # Create dataset
    print("-- Creating dataset..")
    train_dataset = EEDataset(
        encodings=train_encodings, labels=train_tag_ids_offadj)
    val_dataset = EEDataset(encodings=val_encodings,
                            labels=val_tag_ids_offadj)

    return train_dataset, train_tags, val_dataset, val_tags, unique_tags, id2tag


def download_data():

    if os.path.exists("test_data"):
        shutil.rmtree("test_data")
        os.makedirs("test_data")
    else:
        os.makedirs("test_data")

    url = 'https://drive.google.com/uc?id=1Z32tmKPjIVkHm88MWjirdNqPfeUgqfxp'
    output = 'test_data/wnut17train.conll'
    gdown.download(url, output, quiet=False)


if __name__ == "__main__":

    download_data()
    create_dataset(file_path='test_data/wnut17train.conll', verbose=True)

    shutil.rmtree("test_data")
