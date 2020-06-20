import sys
import os
from collections import defaultdict
from SentencePiece import SentencePiece
from math import log
from Lattice import Lattice
import pygtrie
from util import *
from pysuffixarray.core import SuffixArray

import pickle

import subprocess


debug_dir="./debug/"

class UnigramModel:
    """どこまで仕事をするのか
    """

    def __init__(self, argv):
        """ get parameter from argv
        """
        self.debug_cnt=0 #あとで消す。debugのfile名を　書き込まれた順にsortするためにfileのprefixに付ける
        self.file = arg_parser(argv,"file",required=True)
        self.out_voc_file = arg_parser(argv,"voc",required=True)
        self.shrinking_rate = arg_parser(argv,"shrinking_rate",default_val=0.75)
        self.desired_voc_size = arg_parser(argv,"shrinking_rate",default_val=8000)
        self.seed_sentence_piece_size = arg_parser(argv,"seed_sentence_piece_size",default_val=1e5)

        self.use_original_make_seed = arg_parser(argv,"use_original_make_seed",default_val=False)
        self.unk_surface=arg_parser(argv,"unk_surface",default_val="⁇")

        # original spの"_"の太文字みたいな文字
        self.sep_voc = arg_parser(argv,"sep_voc",default_val=chr(9601))

        self.SentencePiece = SentencePiece()
        self.Trie = None
        self.sentences = []
        self.words = []

    def read_sentencenpiece_from_voc_file(self, path):
        """
        trained vocからsentencepiceを読み取って、モデルにセットする
        Arguments:
            path(str): path to trained voc file
        """
        Voc = {}
        with open(path) as f:
            for s in f:
                key, val = s.split("\t")
                Voc[key] = float(val)
        #self.set_sentnece_piece(Voc)
        self.set_sentence_piece(Voc,debug_name="seed",info="init_piece")

    def load_seed_sentencepiece_from_file(self):
        """c++のsentencepieceのmake_seedを呼び出してvocをとってくる。
        unigram_model_trainer.c++のEM前で、seed_pieceを求めた後に、fileに SaveVocab()と似た感じで、fileに書き込んで終了している。
        EMに入る前で止めている。
        original sentence pieceを書き換えないと動かないので注意
        """
        f_name=self.file.split("/")[-1]
        if os.path.isfile(f_name+".seed.vocab"):
            print("seed file is already exsists. skip c++ code")
        else:
            print("run MakeSeedSentence of original c++ sentnecepiece code to get initial piece")
            try:
                res = subprocess.run(["/home/ueki.k/.src/sentencepiece/build/src/spm_train","--input",self.file,"--model_prefix",f_name+".seed","--seed_sentencepiece_size",str(self.seed_sentence_piece_size)])
                print("res=>",res)
            except:
                print("error")
                exit()
            print(res)

        Voc={}
        with open(f_name+".seed.vocab") as f:
            for s in f:
                key,val = s.split("\t")
                Voc[key]=float(val)
        return Voc

    def make_seed_sentence_piece(self):
        """ set init vocabulary of sentence piece

        Return:
            seed_sentencepieces(dict): dict[piece]=score
        """

        all_chars = defaultdict(int)
        array = []

        # Merge all sentences into one array with 0x0000 delimiter
        kSentenceBoundary = chr(0x0000)

        for (word, freq) in self.words.items():
            # ここでpretolenizeってのをかましている
            for c in word:
                uni_c = UTF8ToUnicodeText(c)
                c = UnicodeCharToUTF8(uni_c)
                assert isValidCodepoint(c),"isValidCodepoint {}".format(c)
                assert c!=0x0020,"space must not be included"
                if c==kSentenceBoundary:
                    print("Find null char")
                    continue
                # really needed?
                array.append(c)
                all_chars[c] += freq
            array.append(kSentenceBoundary)

        print("alphabet=>", len(all_chars))

        # make a suffix_array to extract all sub strings occuring more than 2 times in the sentence
        # ここは、配列のサイズによって、分割した方が良さそうな気がする?sa-isでやっているはずで、線形アルゴリズムだから関係ない説もある
        A = "".join(array)
        print("Making Suffix Array len:{}".format(len(array)))
        SA = SuffixArray(A)

        print("Extracting frequent sub strings...")
        # TODO 結構怪しい気がする ここの処理
        substr = set()
        for i, l in enumerate(SA.longest_common_prefix()):
            if l <= 1:  # lcp=1なので1回しか出てこない
                continue
            sb = SA.string[SA.sa[i]:SA.sa[i]+l]  # 2回以上出てくるsbst
            if sb[-1] == kSentenceBoundary:  # 最後の "0x00"は大目に見る
                sb = sb[:-1]
            if len(sb) <= 1:  # 多目に見た後に長さが2.elseはsb=charになっている
                continue
            if any(v == kSentenceBoundary for v in sb):  # 途中に 0x00が入っているのはinvalid
                continue

            if not isValidSentencePiece(sb):
                continue

            # それでも残ったやつは、2回以上出てくるsbst
            freq = len(SA.match(sb))
            assert freq >= 2
            substr.add((sb, len(sb)*freq))

        substr = sorted(list(substr), key=lambda x: -x[1])
        seed_sentencepieces = all_chars
        if len(seed_sentencepieces) > self.seed_sentence_piece_size:
            pass
        elif len(seed_sentencepieces)+len(substr) > self.seed_sentence_piece_size:
            delete_size = len(seed_sentencepieces) + len(substr) -  self.seed_sentence_piece_size
            print(
                "del {} freq-sbst because of seed_sentence_piece_size".format(delete_size))
            for sb, val in substr[:int(delete_size)]:
                seed_sentencepieces[sb] = val
        else:
            for sb, val in substr:
                seed_sentencepieces[sb] = val

        # TO LOG PROB
        s = log(sum([v for v in seed_sentencepieces.values()]))
        for i, v in seed_sentencepieces.items():
            seed_sentencepieces[i] = log(v)-s

        print("Initialized {} seed sentence pieces".format(
            len(seed_sentencepieces)))
        return seed_sentencepieces

    def dump_to_pickle(self,name,data):
        """
        dump data into pickle
        """
        with open(debug_dir+"{:2}_".format(self.debug_cnt)+name+".pickle","wb") as f:
                pickle.dump(data,f)


    def set_sentence_piece(self, pieces,debug_name=None,info=None):
        """ set piece into Sentencepiece class
        Always call build_trie to create new Trie corresponding to new_pieces
        Args:
            pieces(dict): current sentencepieces dict[piece]=score
        """
        if debug_name is not None:
            _, obj_before,_ = self.run_e_step()

            pruned_voc = set(self.SentencePiece.get_pieces().keys() - pieces.keys())

            #pieceの更新
            self.SentencePiece._set_sentence_piece(pieces)
            self.build_trie(pieces)
            ###
            _, obj_after,_ = self.run_e_step()

            debug_info={"obj_before":obj_before,"obj_after":obj_after,"pruned_voc":pruned_voc,"info":info,"gain":obj_before-obj_after}
            self.dump_to_pickle(debug_name,debug_info)

        self.SentencePiece._set_sentence_piece(pieces)
        self.build_trie(pieces)
        self.debug_cnt+=1

    def load_sentence(self,path=None):
        """ load sentence from file
        """
        if path is None:
            path = self.file
        sentences = []
        words = defaultdict(int)
        with open(path) as f:
            for s in f:
                #originalは半角のみを扱っていたので、半角のみを扱うようにする。
                # _s = "_"+"_".join(s.split(" "))#全角と半角のspaceを区別するか(\tとか\nもsplitされるs.split())
                #TODO ここで、lastの"\n"を消すのもあり
                _s = self.sep_voc + self.sep_voc.join(s.split(" "))
                for w in s.split(" "):
                    words[self.sep_voc+w] += 1
                sentences.append(_s)

        self.sentences = sentences
        self.words = words

    def run_e_step(self):
        """E step of EM learning
        Return:
            objective(int): int
            nun_token(int): sum of the token num of Viterbi path
            expected(dict): dict[piece]=score of the piece
        """
        # TODO とりあえず のみ
        expected = defaultdict(int)
        objective = 0
        num_tokens = 0

        all_sentence_freq = sum(self.words.values())

        for key, freq in sorted(self.words.items()):
            L = Lattice()
            L.set_sentence(key)
            L.populate_nodes(self.SentencePiece.get_pieces(), self.Trie)
            Z, ret_expected = L.populate_marginal(freq)

            for key, val in ret_expected.items():
                expected[key] += val

            N = len(L.Viterbi())
            num_tokens += N
            objective -= Z/all_sentence_freq

        return expected, objective, num_tokens

    def run_m_step(self, expected):
        """ M step of EM learning
        Return:
            new_sentencepieces: list of sentencepiece
        """

        assert self.SentencePiece.get_piece_size() >=len(expected),"different:{}".format(expected.keys()-self.SentencePiece.get_pieces().keys())

        new_pieces = dict()
        sum_freq = 0
        kExpectedFrequencyThreshold = 0.5
        # filter infrequent sentencepieces here
        for key, val in self.SentencePiece.get_pieces().items():
            freq = expected[key]
            if freq==0:
                #seed pieceをoriginalから持ってきている場合、expected[piece]がない場合(default dicなので0)になることがある。skipする
                print("cntinue at m step {}".format(key))

            if freq < kExpectedFrequencyThreshold:
                continue
            new_pieces[key] = freq
            sum_freq += freq
        print("M stel filtered infrequent sentencepiece, {} pieces removed".format(
            self.SentencePiece.get_piece_size()-len(new_pieces)))

        logsum = Digamma(sum_freq)
        for key, val in new_pieces.items():
            new_pieces[key] = Digamma(val)-logsum
        return new_pieces

    def prune_step_1_always_keep_alternative(self):
        """
        Return
            always_keep(dict)
            alternatives(dict)
        """
        current_piece = self.SentencePiece.get_pieces()
        # pieceをkeyとしてdictで管理
        always_keep = dict()
        alternatives = defaultdict(list)

        # First segments the current sentencepieces to kwon how each sentencepiece is resegmented if this sentencepiece is  removed from vocabulary.
        for key, score in current_piece.items():
            L = Lattice()
            L.set_sentence(key)
            L.populate_nodes(current_piece, self.Trie)
            nbests = L.NBest(2, ret_piece=True)

            if len(nbests) == 1:  # only one way to resegment it
                always_keep[key] = True

            elif len(nbests[0]) >= 2:
                always_keep[key] = False

            elif len(nbests[0]) == 1:
                always_keep[key] = True
                alternatives[key] = nbests[1]

        return always_keep, alternatives

    def prune_step_2_freq_inverted(self):
        """
        Return
            vsum(float):
            freq(dict):
            inverted(dict):
        """
        current_piece = self.SentencePiece.get_pieces()
        vsum = 0
        freq = defaultdict(int)
        # inverted[key] stires the set of sentence index where the sentencepiece (key) appears
        inverted = defaultdict(int)

        for s, score in self.words.items():
            vsum += score
            L = Lattice()
            L.set_sentence(s)
            L.populate_nodes(current_piece, self.Trie)

            for word in L.Viterbi(ret_piece=True):
                freq[word] += score
                inverted[word] += score

            # remove this
            for node_id in L.Viterbi():
                word = L.nodes[node_id].piece
                if node_id > 0:
                    # TODO what is difference of freq and inverted
                    #freq[word] += score
                    # inverted[word]+=score
                    pass
                else:
                    print("prune2=>", word)

        return vsum, freq, inverted

    def prune_step_3_new_piece_cand(self, always_keep, alternatives, vsum, freq, inverted):
        """
        Return
            candiate[]: candidate of new pieces
            new_sentencepieces(dict):
        """
        sum_freq = sum(freq.values())
        logsum = log(sum_freq)

        candidate = dict()
        new_sentencepieces = dict()

        for key, val in self.SentencePiece.get_pieces().items():
            if freq[key] == 0 or not always_keep[key]:
                continue
            elif len(alternatives[key]) == 0:
                new_sentencepieces[key] = val
            else:
                F = inverted[key]
                F /= vsum  # keyが出てくる文の数を全文数で割ったもの
                # keyの出現確率( P(x)= \frac{freq_x}{sum(all_piece_freq)})
                logprob_sp = log(freq[key])-logsum
                # x->x_altに置換後の log(freq_sum)
                logsum_alt = log(sum_freq+freq[key]*(len(alternatives)-1))

                logprob_alt = 0
                for alt in alternatives[key]:
                    logprob_alt += (log(freq[alt]+freq[key])-logsum_alt)

                # Freq*(logp(x)-logp(x_alts))
                #(P(X)よりP(x_alt)の方が高いとき、logp(x)= -大 logp(x_alt)=-小->loss=-大
                #P(x)が小さい場合->pieceを分けた方がいい。
                loss = F*(logprob_sp-logprob_alt)
                candidate[key] = loss

        return candidate, new_sentencepieces

    def prune_4_prune_candidate(self, candidate, new_sentencepieces):
        """
        Return
            candidate(dict): dict[key] = loss of key
            new_sentencepieces(dict):
        """
        assert  len(new_sentencepieces)<=self.desired_voc_size,"{}".format(len(new_sentencepieces))
        current_piece = self.SentencePiece.get_pieces()
        pruned_size = max(
            int(len(current_piece)*self.shrinking_rate), self.desired_voc_size)

        candidate_list = [(key, val) for key, val in candidate.items()]
        #for piece, _ in sorted(candidate_list, key=lambda x: x[1], reverse=True):
        #lossはp(x)<p(x_alt)の時に、-大(xを分割したいとき),逆に loss=が大きい時は、p(x)を残した方がいいとき。よって、sorted(reverse)
        for piece, _ in sorted(candidate_list, key=lambda x: x[1], reverse=True):
            # add piece from candidate in decsengind order of score till piece size reaches to pruned_size
            if len(new_sentencepieces) == pruned_size:
                break
            new_sentencepieces[piece] = current_piece[piece]
        print("prune step {} pieces are pruned".format(
            len(current_piece) - len(new_sentencepieces)))
        #assert len(current_piece)==len(new_sentencepieces),"no piece is  pruned"
        return new_sentencepieces

    def prune_piece(self):
        # First,
        always_keep, alternatives = self.prune_step_1_always_keep_alternative()
        # Second, segments all sentences to compute likelihoood with a Unigram LM
        vsum, freq, inverted = self.prune_step_2_freq_inverted()
        # Third
        candidate, new_sentencepieces = self.prune_step_3_new_piece_cand(
            always_keep, alternatives, vsum, freq, inverted)
        # Forth,
        new_sentencepieces = self.prune_4_prune_candidate(
            candidate, new_sentencepieces)

        assert self.SentencePiece.get_piece_size()!=len(new_sentencepieces),"no piece is  pruned"
        return new_sentencepieces

    def finalize_sentencepiece(self):
        """最終的な処理
        fileへの書き込みをする
        """
        print("finally, {} pieces".format(self.SentencePiece.get_piece_size()))
        piece = self.SentencePiece.get_pieces()
        with open(self.out_voc_file, "w") as f:
            for key, val in sorted(piece.items(), key=lambda x: -x[1]):
                f.write("{}\t{}\n".format(key, val))
        print("written voc to {}".format(self.out_voc_file))

    def build_trie(self, pieces):
        """ building Trie from piece
        """
        Trie = pygtrie.Trie()
        for (key, score) in pieces.items():
            Trie[key] = (key, score)
        self.Trie = Trie

    def make_seed(self):
        if self.use_original_make_seed:
            seed_sentencepieces = self.load_seed_sentencepiece_from_file()
        else:
            seed_sentencepieces = self.make_seed_sentence_piece()
        return seed_sentencepieces

    def train(self):
        """ training 
        """
        self.load_sentence()
        seed_sentencepieces = self.make_seed()
        self.set_sentence_piece(seed_sentencepieces)
        #self.set_sentence_piece(seed_sentencepieces,debug_name="train_start")

        step_cnt = 0
        print("init vocab size is {}\n start EM trainind".format(self.SentencePiece.get_piece_size()))
        while True:
            step_cnt += 1
            for itr in range(2):  # EM iteration loop
                expected, objective, num_tokens = self.run_e_step()
                new_sentencepieces = self.run_m_step(expected)

                self.set_sentence_piece(new_sentencepieces,debug_name="step{}_mstep{}".format(step_cnt,itr))

                piece_size = self.SentencePiece.get_piece_size()
                print("EM sub_iter= {} size={} obj={} num_tokens= {} num_tokens/piece= {}".format(
                    itr, piece_size, objective, num_tokens, num_tokens/piece_size))

            if len(new_sentencepieces) <= self.desired_voc_size:
                break
            new_sentencepieces = self.prune_piece()
            self.set_sentence_piece(new_sentencepieces,debug_name="step{}_prune".format(step_cnt))

        # Save to file
        print("{} step is needed to converge".format(step_cnt))
        self.finalize_sentencepiece()

    def encode_one_sent(self, sent):
        """
        Arguments:
            sent(str): sentence piece vocを使って分割する文
        Returns:
            tokenize_sent(str): space split tokenize sentence
        """
        L = Lattice()
        L.set_sentence(sent)
        L.populate_nodes(self.SentencePiece.get_pieces(), self.Trie)
        tokenize_sent = " ".join(L.Viterbi(ret_piece=True))
        assert "".join(tokenize_sent.split(" "))==sent
        return tokenize_sent

    def encode(self):
        """
        self.sentencesを全てencode_one()して、listにしてreturn?
        Returns:
            encode_sentences(list):
        """
        encode_sentences = [self.encode_one_sent(s) for s in self.sentences]
        return encode_sentences

    def encode_new(self,path,voc_file):
        """
        pathのぶんをencodeする
        """
        #TODO こいつを使えるようにする
        self.load_sentence(path=path)
        self.read_sentencenpiece_from_voc_file(voc_file)

        ret=[]
        with open(path) as f:
            for s in f:
                if s[-1]=="\n":
                    s = s[:-1]
                encoded_sent = self.encode_one_sent(s)
                ret.append(encoded_sent)
        return ret

        

    def decode_one_piece(self,piece:str):
        """
        PiecegがvocにないならUNKにする。
        Arguments:
            pieceをdecodeする。(UNKに置き換えるやつ)
        Returns:
            piece
        """
        if piece==" ":
            return ""
        if piece in self.SentencePiece.get_pieces().keys():
            return self.unk_surface
        else:
            return piece

    def decode_one_sent(self, s:str)->str:
        """
        Arguments:
            tokenized_sent(str): tokenizeされた文
        Returns:
            ret_sent(str): space split tokenize sentence
        """
        s = "".join(s.split(" ")) #remove whitespace
        s = s.replace("\n","")
        s = s.replace(self.sep_voc," ")
        if len(s)>0 and s[0]==" ": #remove white space located at the beginning of sent(beggining "_")
            s=s[1:]
        return s

    def decode_sent(self,path:str)->list:
        """
        Arguments: decodeしたい文のpath
        """
        #TODO encodeとinterfaceが違うのが気になること
        #TODO 1 or 2. 1:encode_sent(path)? 2. decode_sent(),self.sent
        ret=[]

        with open(path) as f:
            for s in f:
                ret.append(self.decode_one_sent(s))
        return ret




# sample
if __name__ == "__main__":
    arg = {
        "file": "../test/dummy2.en",
        "voc": "dummy.en.voc",
        "seed_sentence_piece_size": 1e5
    }
    U = UnigramModel(arg)
    U.train()
