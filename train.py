from prep_data import create_dataset, download_data
import numpy as np
import os
from ast import literal_eval
import shutil

from transformers import DistilBertForTokenClassification, Trainer, TrainingArguments
from torch import nn
from seqeval.metrics import f1_score, classification_report

# ignore warnings
import warnings
warnings.filterwarnings("ignore")


def compute_metrics(pred):

    f = open("_temp/id2tag.txt", "r")
    content = f.read()
    f.close()

    id2tag = literal_eval(content)

    label_ids = pred.label_ids
    preds = np.argmax(pred.predictions, axis=2)
    batch_size, seq_len = preds.shape

    preds_list = [[] for _ in range(batch_size)]
    label_list = [[] for _ in range(batch_size)]
    for i in range(batch_size):
        for j in range(seq_len):
            # ignore pad_tokens
            if label_ids[i, j] != nn.CrossEntropyLoss().ignore_index:
                preds_list[i].append(id2tag[preds[i][j]])
                label_list[i].append(id2tag[label_ids[i][j]])

    results = dict()

    f1 = f1_score(label_list, preds_list)
    results["f1"] = f1

    rep = classification_report(label_list, preds_list).split('\n')
    par_names = rep[0].split(" ")
    par_names = [x for x in par_names if x != '']

    rep = rep[2:(len(rep)-4)]
    for r in rep:
        r = r.split(" ")
        r = [x for x in r if x != '']
        ee_name = r[0]
        r = r[1:]

        for idx, par_name in enumerate(par_names):
            results[ee_name+"_"+par_name] = float(r[idx])

    return results


def main(file_path):
    train_dataset, train_tags, val_dataset, val_tags, unique_tags, id2tag = create_dataset(
        file_path=file_path)

    data = train_dataset, train_tags, val_dataset, val_tags, unique_tags, id2tag

    if os.path.exists("_temp"):
        shutil.rmtree("_temp")

    os.makedirs("_temp")
    f = open("_temp/id2tag.txt", "w")
    f.write(str(id2tag))
    f.close()

    print("-- Loading model..")
    model = DistilBertForTokenClassification.from_pretrained(
        'distilbert-base-cased', num_labels=len(unique_tags))

    training_args = TrainingArguments(
        output_dir='./results',          # output directory
        num_train_epochs=3,              # total number of training epochs
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=64,   # bxatch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        evaluate_during_training=True,
        logging_dir='./logs',            # directory for storing logs
        logging_first_step=True,
        logging_steps=250,
        eval_steps=250,
        save_steps=1000
    )

    trainer = Trainer(
        # the instantiated 🤗 Transformers model to be trained
        model=model,
        args=training_args,                  # training arguments, defined above
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,         # training dataset
        eval_dataset=val_dataset             # evaluation dataset
    )

    print("-- Start training..")
    trainer.train()

    #print("-- Start evaluation.. ")
    # trainer.evaluate()

    shutil.rmtree("_temp")
    return trainer, data


if __name__ == "__main__":

    download_data()

    FILE_PATH = 'test_data/wnut17train.conll'
    trainer, data = main(file_path=FILE_PATH)
    shutil.rmtree("test_data")
