import sys
#sys.path.append("../src/")
from UnigramModel import UnigramModel

def Encode(corpus_path,voc_file_path, n_threads=1):
    dummy_arg = {
        "file": corpus_path,
        "voc": None,
        "vocab_size":8000,
        "shrinking_rate": None,
        "desired_voc_size": None,
        "seed_sentence_piece_size":None,
        "use_original_make_seed":False,
        "n_threads":int(n_threads),
    }

    U = UnigramModel(dummy_arg)
    U.read_sentencenpiece_from_voc_file(voc_file_path)
    encoded_sents = U.encode(corpus_path)
    for i in range(len(encoded_sents)):
        if encoded_sents[i][-1]=="\n":
            encoded_sents[i] = encoded_sents[i][:-1]
    print("\n".join(encoded_sents))

if __name__=="__main__":
    argv=sys.argv
    assert len(argv)>=3,"requiereed 2 params :: corpus_path,voc_file_path"
    Encode(*argv[1:])



