import sys
sys.path.append("../src/")
from UnigramTrainer import Train


def Train_En():
    print("Train EN")
    arg = {
        "file": "../corpus/train50K.en",
        "voc": "../test50K/voc/separate.en.voc",
        "use_original_make_seed":True,
        "shrinking_rate": 0.75,
        "desired_voc_size": 8000,
        "seed_sentence_piece_size": 1e5
    }
    Train(arg)


def Train_Ja():
    print("Train JA")
    arg = {
        "file":"../corpus/train50K.jap",
        "voc": "../test50K/voc/separate.jap.voc",
        "use_original_make_seed":True,
        "shrinking_rate": 0.75,
        "desired_voc_size": 8000,
        "seed_sentence_piece_size": 1e5
    }
    Train(arg)


if __name__ == "__main__":
    Train_En()
    Train_Ja()
