# lattice とUnigramをimport するために必要
import sys
sys.path.append("../src/")
from UnigramTrainerAlign import train_align


debug=True
debug_dir="./debug_alpha1_no_approx_words_en300/"
if __name__ == "__main__":
    arg_mini_en = {
        "file": "../corpus/mini.en",
        "voc":debug_dir+"en.voc",
        "voc": "./res_voc/dummy.en.voc",
        "use_original_make_seed":True,
        "vocab_size":300,
        "debug":debug,
        "debug_dir":debug_dir,
    }
    arg_mini_ja = {
        "file": "../corpus/mini.jap",
        "voc":debug_dir+"jap.voc",
        "use_original_make_seed":True,
        "debug":debug,
        "vocab_size":1500,
        "debug_dir":debug_dir,
    }
    train_align(arg_mini_en,arg_mini_ja,debug=debug,alpha=1)
