# lattice とUnigramをimport するために必要
import sys
sys.path.append("../src/")
from UnigramTrainerAlign import train_align

if __name__ == "__main__":
    arg_mini_en = {
        "file": "../corpus/train5K.en",
        "voc": "./res_voc/dummy.en.voc",
        "use_original_make_seed":True,
    }
    arg_mini_ja = {
        "file": "../corpus/train5K.jap",
        "voc": "./res_voc/dummy.jap.voc",
        "use_original_make_seed":True,
    }
    train_align(arg_mini_en,arg_mini_ja)
