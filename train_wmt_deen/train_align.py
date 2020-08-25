# lattice とUnigramをimport するために必要
import sys
sys.path.append("../src/")
from UnigramTrainerAlign import train_align

DIR="../../data/wmtEnDe/"
VOC_DST="../../align_tokenize_deen/vocs/"
BACK_UP="../../align_tokenize_deen/backup/backup"

if __name__ == "__main__":
    arg_en = {
        "file":DIR+"train.en",
        "voc":VOC_DST+"align001.en",
        "use_original_make_seed":True,
        "shrinking_rate": 0.75,
        "vocab_size":8000,
    }
    arg_de = {
        "file":DIR+"train.de",
        "voc":VOC_DST+"align001.de",
        "use_original_make_seed":True,
        "shrinking_rate": 0.75,
        "vocab_size":8000,
    }
    train_align(arg_en,arg_de,alpha=0.01,back_up_interval=3,back_up_file=BACK_UP)
