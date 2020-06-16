# lattice とUnigramをimport するために必要
import sys
sys.path.append("../src/")
from UnigramTrainerAlign import train_align

if __name__ == "__main__":
    arg_en_allA= {
        "file": "../corpus/train50K.en",
        "voc": "../test50K/voc/align.allA.en.voc",
        "shrinking_rate": 0.75,
        "use_original_make_seed":True,
        "desired_voc_size": 8000,
        "seed_sentence_piece_size": 1e5
    }
    arg_ja_allA = {
        "file": "../corpus/train50K.jap",
        "voc": "../test50K/voc/align.allA.jap.voc",
        "shrinking_rate": 0.75,
        "use_original_make_seed":True,
        "desired_voc_size": 8000,
        "seed_sentence_piece_size": 1e5
    }
    train_align(arg_en_allA,arg_ja_allA,allA=True)
