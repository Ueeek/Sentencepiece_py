
import sys
sys.path.append("../../src")
from UnigramModel import UnigramModel

def Decode(corpus_path):
    arg = {
        "file": corpus_path,
        "voc": None,
        "shrinking_rate": None,
        "desired_voc_size": None,
        "seed_sentence_piece_size":None,
        "use_original_make_seed":False,
    }
    U = UnigramModel(arg)
    ret = U.decode_sent(corpus_path)
    print("\n".join(ret))

if __name__=="__main__":
    argv=sys.argv
    assert len(argv)==2,"requiereed 2 params :: corpus_path"
    Decode(*argv[1:])



