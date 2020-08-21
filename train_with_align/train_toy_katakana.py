# lattice とUnigramをimport するために必要
import sys
sys.path.append("../src/")
from UnigramTrainerAlign import train_align

if __name__ == "__main__":
    arg_en = {
        "file":"../../data/edict/katakana.train.en",
        "voc":"../../toy_katakana_translation/vocs/align.alpha001.800.en",
        "use_original_make_seed":True,
        "shrinking_rate": 0.75,
        "vocab_size":800,
    }
    arg_ja = {
        "file":"../../data/edict/katakana.train.ja",
        "voc":"../../toy_katakana_translation/vocs/align.alpha001.800.ja",
        "use_original_make_seed":True,
        "shrinking_rate": 0.75,
        "vocab_size":800,
    }
    train_align(arg_en,arg_ja,alpha=0.01)
