# lattice とUnigramをimport するために必要
import sys
sys.path.append("../src/")
from UnigramTrainerAlign import train_align

if __name__ == "__main__":
    arg_mini_en = {
        "file": "../corpus/test5K.mini.en",
        "voc": "./res_voc/dummy.en.voc",
        "shrinking_rate": 0.75,
        "desired_voc_size": 90,
        "seed_sentence_piece_size": 1e5
    }
    arg_mini_ja = {
        "file": "./corpus/test.mini.jap",
        "voc": "./res_voc/dummy.jap.voc",
        "shrinking_rate": 0.75,
        "use_original_make_seed":True,
        "desired_voc_size": 90,
        "seed_sentence_piece_size": 1e5
    }
    arg_en = {
        "file": "../corpus/train.en",
        "voc": "./res_voc/align.en.voc",
        "use_original_make_seed":True,
        "shrinking_rate": 0.75,
        "desired_voc_size": 8000,
        "seed_sentence_piece_size": 1e5
    }
    arg_ja = {
        "file": "../corpus/train.jap",
        "voc": "./res_voc/align.jap.voc",
        "use_original_make_seed":True,
        "shrinking_rate": 0.75,
        "desired_voc_size": 8000,
        "seed_sentence_piece_size": 1e5
    }
    arg_en_allA= {
        "file": "../corpus/train.en",
        "voc": "./res_voc/align.allA.en.voc",
        "shrinking_rate": 0.75,
        "use_original_make_seed":True,
        "desired_voc_size": 8000,
        "seed_sentence_piece_size": 1e5
    }
    arg_ja_allA = {
        "file": "../corpus/train.jap",
        "voc": "./res_voc/align.allA.jap.voc",
        "shrinking_rate": 0.75,
        "use_original_make_seed":True,
        "desired_voc_size": 8000,
        "seed_sentence_piece_size": 1e5
    }
    arg_en_alter = {
        "file": "../corpus/train.en",
        "voc": "./res_voc/align.alter.en.voc",
        "shrinking_rate": 0.75,
        "desired_voc_size": 8000,
        "use_original_make_seed":True,
        "seed_sentence_piece_size": 1e5
    }
    arg_ja_alter= {
        "file": "../corpus/train.jap",
        "voc": "./res_voc/align.alter.jap.voc",
        "shrinking_rate": 0.75,
        "use_original_make_seed":True,
        "desired_voc_size": 8000,
        "seed_sentence_piece_size": 1e5
    }
    #train_align(arg_mini_en,arg_mini_ja,alter=True)
    #train_align(arg_en,arg_ja)
    train_align(arg_en_allA,arg_ja_allA,allA=True)
    #train_align(arg_en_alter,arg_ja_alter,alter=True)
