# lattice とUnigramをimport するために必要
import sys
sys.path.append("../src/")
from UnigramTrainerAlign import train_align

DIR="../../data/wmtEnDe/"
VOC_DST="../../align_tokenize_deen/vocs/"

if __name__ == "__main__":
    arg_en = {
        "file":DIR+"train.en",
        "voc":VOC_DST+"align001.en",
        "use_original_make_seed":True,
        "shrinking_rate": 0.75,
        "vocab_size":8000,
    }
    arg_ja = {
        "file":DIR+"train.de",
        "voc":VOC_DST+"align001.de",
        "use_original_make_seed":True,
        "shrinking_rate": 0.75,
        "vocab_size":800,
    }
    train_align(arg_en,arg_ja,alpha=0.01)
