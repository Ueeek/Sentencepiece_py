import sys
from UnigramModel import UnigramModel


import io

sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding="utf-8")

def Decode(corpus_path,voc_file):
    dummy_arg = {
        "file": corpus_path,
        "voc": None,
        "vocab_size":9,
        "shrinking_rate": None,
        "desired_voc_size": None,
        "seed_sentence_piece_size":None,
        "use_original_make_seed":False,
    }
    U = UnigramModel(dummy_arg)
    U.load_voc_file(voc_file)
    ret = U.decode_sent(corpus_path)
    print("\n".join(ret))

if __name__=="__main__":
    argv=sys.argv
    assert len(argv)==3,"requiereed 2 params :: corpus_path, voc"
    Decode(*argv[1:])



