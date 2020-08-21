import sys
sys.path.append("../src/")
from UnigramTrainer import Train


DIR="../../data/wmtEnDe/"
VOC_DST="../../align_tokenize_deen/vocs/"

def Train_En():
    print("Train EN")
    arg = {
        "file":DIR+"train.en",
        "voc":VOC_DST+"separete.en.voc",
        "shrinking_rate": 0.75,
        "use_original_make_seed":True,
        "vocab_size":8000,
        "seed_sentence_piece_size": 1e5
    }
    Train(arg)


def Train_De():
    print("Train DE")
    arg = {
        "file":DIR+"train.de",
        "voc":VOC_DST+"separete.de.voc",
        "shrinking_rate": 0.75,
        "use_original_make_seed":True,
        "vocab_size":8000,
        "seed_sentence_piece_size": 1e5
    }
    Train(arg)


if __name__ == "__main__":
    Train_En()
    Train_De()
