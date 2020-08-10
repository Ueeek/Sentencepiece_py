# lattice とUnigramをimport するために必要
import sys
sys.path.append("../src/")
from UnigramTrainerAlign import train_align

if __name__ == "__main__":
    arg_en = {
        "file": "../..//data/edict/toycorpus_kanji.txt.en",
        "voc":"../toydata_large/voc/align001.kanji.1000.en.voc",
        "use_original_make_seed":True,
        "shrinking_rate": 0.75,
        "vocab_size":1000,
    }
    arg_ja = {
        "file": "../..//data/edict/toycorpus_kanji.txt.ja",
        "voc":"../toydata_large/voc/align001.kanji.1500.ja.voc",
        "use_original_make_seed":True,
        "shrinking_rate": 0.75,
        "vocab_size":1500,
    }
    train_align(arg_en,arg_ja,alpha=0.01)
