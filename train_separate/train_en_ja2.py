import sys
sys.path.append("../src/")
from UnigramTrainer import Train


def Train_En():
    print("Train EN")
    arg = {
        "file": "../corpus/train.en",
        "voc":"../testFull/voc/separate.en.voc",
        "shrinking_rate": 0.75,
        "use_original_make_seed":True,
        "vocab_size":8000,
        "seed_sentence_piece_size": 1e5
    }
    Train(arg)


def Train_Ja():
    print("Train JA")
    arg = {
        "file":"../corpus/train.jap",
        "voc":"../testFull/voc/separate.jap.voc",
        "shrinking_rate": 0.75,
        "use_original_make_seed":True,
        "vocab_size":8000,
        "seed_sentence_piece_size": 1e5
    }
    Train(arg)


if __name__ == "__main__":
    Train_En()
    Train_Ja()
