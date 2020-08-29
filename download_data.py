import os
import wget

if __name__ == "__main__":
    if not os.path.exists("wnut17train.conll"):
        wget.download(
            "http://noisy-text.github.io/2017/files/wnut17train.conll")
