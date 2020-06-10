import sys
sys.path.append("../../src")
from UnigramModel import UnigramModel

def Encode(corpus_path,voc_file_path):
    arg = {
        "file": corpus_path,
        "voc": None,
        "shrinking_rate": None,
        "desired_voc_size": None,
        "seed_sentence_piece_size":None,
    }
    U = UnigramModel(arg)
    U.load_sentence()
    U.read_sentencenpiece_from_voc_file(voc_file_path)

    encoded_sents = U.encode()
    print("\n".join(encoded_sents))

if __name__=="__main__":
    argv=sys.argv
    assert len(argv)==3
    Encode(*argv[1:])



