# lattice とUnigramをimport するために必要
import sys
sys.path.append("../src/")
from UnigramTrainerAlign import train_align


debug=True
debug_dir="./debug_alpha001/"
if __name__ == "__main__":
    arg_mini_en = {
        "file": "../corpus/train5K.en",
        "voc": "./res_voc/dummy.en.voc",
        "use_original_make_seed":True,
        "vocab_size":4000,
        "debug":debug,
        "debug_dir":debug_dir
    }
    arg_mini_ja = {
        "file": "../corpus/train5K.jap",
        "voc": "./res_voc/dummy.jap.voc",
        "use_original_make_seed":True,
        "debug":debug,
        "vocab_size":4000,
        "debug_dir":debug_dir
    }
    train_align(arg_mini_en,arg_mini_ja,alpha=1,debug=debug)
