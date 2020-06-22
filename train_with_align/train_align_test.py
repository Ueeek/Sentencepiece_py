# lattice とUnigramをimport するために必要
import sys
sys.path.append("../src/")
from UnigramTrainerAlign import train_align


debug=False
if __name__ == "__main__":
    arg_mini_en = {
        "file": "../corpus/mini.en",
        "voc": "./res_voc/dummy.en.voc",
        "use_original_make_seed":True,
        "debug":debug,
        "desired_voc_size":1500,
    }
    arg_mini_ja = {
        "file": "../corpus/mini.jap",
        "voc": "./res_voc/dummy.jap.voc",
        "use_original_make_seed":True,
        "debug":debug,
        "desired_voc_size":1500,
    }
    train_align(arg_mini_en,arg_mini_ja,debug=debug)
