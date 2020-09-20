import sys
from UnigramModel import UnigramModel

def Decode(corpus_path,voc_path):
    dummy_arg = {
        "file": corpus_path,
        "vocab_size":8000,
        "voc": None,
        "shrinking_rate": None,
        "desired_voc_size": None,
        "seed_sentence_piece_size":None,
        "use_original_make_seed":False,
    }
    U = UnigramModel(dummy_arg)
    U.load_voc_file(voc_path)
    ret = U.decode_sent(corpus_path)
    print("\n".join(ret))

if __name__=="__main__":
    argv=sys.argv
    assert len(argv)==3,"requiereed 2 params :: corpus_path ,voc_path given:{}".format(argv[1:])
    Decode(*argv[1:])



