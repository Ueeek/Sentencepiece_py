# lattice とUnigramをimport するために必要
import sys
sys.path.append("../src/")
from UnigramTrainerAlign import train_align


debug=True
if __name__ == "__main__":
    arg_mini_en = {
        "file": "../corpus/train5K.en",
        "voc": "./res_voc/dummy.en.voc",
        "use_original_make_seed":True,
        "vocab_size":4000,
        "debug":debug,
    }
    arg_mini_ja = {
        "file": "../corpus/train5K.jap",
        "voc": "./res_voc/dummy.jap.voc",
        "use_original_make_seed":True,
        "debug":debug,
        "vocab_size":4000,
    }
    train_align(arg_mini_en,arg_mini_ja,debug=debug)
