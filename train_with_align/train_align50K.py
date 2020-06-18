# lattice とUnigramをimport するために必要
import sys
sys.path.append("../src/")
from UnigramTrainerAlign import train_align

if __name__ == "__main__":
    arg_en = {
        "file": "../corpus/train50K.en",
        "voc": "../test50K/voc/align.en.voc",
        "use_original_make_seed":True,
        "shrinking_rate": 0.75,
        "desired_voc_size": 8000,
        "seed_sentence_piece_size": 1e5
    }
    arg_ja = {
        "file": "../corpus/train50K.jap",
        "voc": "../test50K/voc/align.jap.voc",
        "use_original_make_seed":True,
        "shrinking_rate": 0.75,
        "desired_voc_size": 8000,
        "seed_sentence_piece_size": 1e5
    }
    train_align(arg_en,arg_ja)
