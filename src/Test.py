# lattice とUnigramをimport するために必要
from math import log,exp
from Lattice import Lattice
from UnigramModel import UnigramModel
from collections import defaultdict
import pickle

class Test:
    def __init__(self,arg_src, arg_tgt):

        self.U_src = UnigramModel(arg_src)
        self.U_tgt = UnigramModel(arg_tgt)
        print("init")
        input()
    def prepare_UnigramModel(self):
        # load sentence
        print("load_sentence")
        self.U_src.load_sentence()
        self.U_tgt.load_sentence()
        # seed_piece
        print("make seed")

        seed_src = self.U_src.make_seed()
        seed_tgt = self.U_tgt.make_seed()

        self.U_src.set_sentence_piece(seed_src)
        self.U_tgt.set_sentence_piece(seed_tgt)
