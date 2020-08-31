# lattice とUnigramをimport するために必要
from nltk.translate import IBMModel1
from nltk.translate import IBMModel
from nltk.translate import Alignment
from nltk.translate import AlignedSent
from math import log,exp
from Lattice import Lattice
from UnigramModel import UnigramModel
from collections import defaultdict
import pickle


DIR="../../data/wmtEnDe/"
VOC_DST="../../align_tokenize_deen/vocs/"
arg_src = {
    "file":DIR+"train.en",
    "voc":VOC_DST+"test.en",
    "shrinking_rate": 0.75,
    "use_original_make_seed":True,
    "vocab_size":8000,
    "seed_sentence_piece_size": 1e5
}
arg_tgt = {
    "file":DIR+"train.de",
    "voc":VOC_DST+"test.de",
    "shrinking_rate": 0.75,
    "use_original_make_seed":True,
    "vocab_size":8000,
    "seed_sentence_piece_size": 1e5
}

u_src = UnigramModel(arg_src)
u_tgt = UnigramModel(arg_tgt)
