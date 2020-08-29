from transformers import DistilBertForTokenClassification, Trainer, TrainingArguments
from prep_data import create_dataset

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


def main(file_path):
    train_dataset, train_tags, val_dataset, val_tags, unique_tags, id2tag = create_dataset(
        file_path=file_path)

    data = train_dataset, train_tags, val_dataset, val_tags, unique_tags, id2tag

    print("-- Loading model..")
    model = DistilBertForTokenClassification.from_pretrained(
        'distilbert-base-cased', num_labels=len(unique_tags))

    training_args = TrainingArguments(
        output_dir='./results',          # output directory
        num_train_epochs=20,              # total number of training epochs
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=64,   # bxatch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        logging_steps=1,
    )

    trainer = Trainer(
        # the instantiated 🤗 Transformers model to be trained
        model=model,
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=val_dataset             # evaluation dataset
    )

    print("-- Start training..")
    trainer.train()

    return trainer, data


if __name__ == "__main__":
    main(file_path='test_data/wnut17train.conll')
