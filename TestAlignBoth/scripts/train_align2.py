import sys
sys.path.append("../../src")

from AlignTrainerBoth import AlignTrainerBoth

SRC="../../../data/kftt-data-1.0/data/orig/kyoto-dev.en"
TGT="../../../data/kftt-data-1.0/data/orig/kyoto-dev.ja"

VOC_SIZE=2000

arg_en = {
    "file": SRC,
    "vocab_size":VOC_SIZE,
    "voc":"../vocs/alignBoth.voc.en",
    "shrinking_rate": 0.75,
    "use_original_make_seed":True,
}
arg_ja = {
    "file": TGT,
    "vocab_size":VOC_SIZE,
    "voc":"../vocs/alignBoth.voc.ja",
    "shrinking_rate": 0.75,
    "use_original_make_seed":True,
}

Model = AlignTrainerBoth(arg_en,arg_ja)
#Model.train_align(alpha=0.01)
Model.train(alpha=0.01)
