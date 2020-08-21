
import sys
sys.path.append("../../src")
from UnigramModel import UnigramModel

def Decode(corpus_path,vocs):
    arg = {
        "voc":"tmp",
        "file": corpus_path,
        "vocab_size": 1000,
        "shrinking_rate": None,
        "desired_voc_size": None,
        "seed_sentence_piece_size":None,
        "use_original_make_seed":False,
    }
    U = UnigramModel(arg)
    U.load_voc_file(vocs)
    ret = U.decode_sent(corpus_path)
    print("\n".join(ret))

if __name__=="__main__":
    argv=sys.argv
    assert len(argv)==3,"requiereed 2 params :: corpus_path.vocs"
    Decode(*argv[1:])



