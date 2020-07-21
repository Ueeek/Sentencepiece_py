# lattice とUnigramをimport するために必要
import sys
sys.path.append("../src/")
from UnigramTrainerAlign import train_align

if __name__ == "__main__":
    arg_en = {
        "file": "../corpus/toycorpus.txt.en",
        "voc": "./test.txt.en",
        #"voc":"../toydata/voc/align.alpha001.en.voc",
        "use_original_make_seed":True,
        "shrinking_rate": 0.75,
        "vocab_size":200,
    }
    arg_ja = {
        "file": "../corpus/toycorpus.txt.ja",
        "voc": "./test.txt.ja",
        #"voc":"../toydata/voc/align.alpha001.jap.voc",
        "use_original_make_seed":True,
        "shrinking_rate": 0.75,
        "vocab_size":600,
    }
    train_align(arg_en,arg_ja,alpha=0.01)
