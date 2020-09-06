import json
import shutils
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


def read_doccano_file(filename="data/file.json1"):
    file = open(filename, "rb")
    lines = file.readlines()

    token_docs = []
    tag_docs = []

    for line in lines:
        text = json.loads(line)["text"]

        labels = json.loads(line)["labels"]
        try:
            lab = labels[0]  # assumes that there is only 1 entity, fix later.

            start_pos = lab[0]
            end_pos = lab[1]
            ent_name = lab[2]

            text_left = text[:start_pos]
            text_entity = text[start_pos:end_pos]
            text_right = text[end_pos:]

            tokens = [text_left, text_entity, text_right]
            tags = ["O", ent_name, "O"]
            token_docs.append(tokens)
            tag_docs.append(tags)

        except:
            print("Couldnt process entities {}".format(labels))

    print(tag_docs)


if __name__ == "__main__":
    read_doccano_file(filename="data/file.json1")
