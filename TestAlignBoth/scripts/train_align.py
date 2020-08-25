import sys
sys.path.append("../../src/")

from UnigramTrainerAlign import train_align

SRC="../../../data/kftt-data-1.0/data/orig/kyoto-dev.en"
TGT="../../../data/kftt-data-1.0/data/orig/kyoto-dev.ja"

VOC_SIZE=2000

arg_en = {
    "file": SRC,
    "vocab_size":VOC_SIZE,
    "voc":"../vocs/align.voc.en",
    "shrinking_rate": 0.75,
    "use_original_make_seed":True,
}
arg_ja = {
    "file": TGT,
    "vocab_size":VOC_SIZE,
    "voc":"../vocs/align.voc.ja",
    "shrinking_rate": 0.75,
    "use_original_make_seed":True,
}

train_align(arg_en,arg_ja,alpha=0.01)
