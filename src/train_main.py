import sys
from UnigramModel import UnigramModel

def main(argv):
    """ start train from this function.call train()

    Args:
    argv:
    """

    model = UnigramModel(argv)
    model.train()

if __name__=="__main__":
    argv = sys.argv
    main(argv[1:])






