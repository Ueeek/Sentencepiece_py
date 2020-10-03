import sys
import os
import time
from collections import defaultdict
from SentencePiece import SentencePiece
from math import log
from Lattice import Lattice
import pygtrie
from util import *
from multiprocessing import Pool
from itertools import zip_longest

from pysuffixarray.core import SuffixArray
import gc


import pickle

import subprocess


class UnigramModel:
    """
    """

    def __init__(self, argv):
        """ get parameter from argv
        """
        if "help" in argv.keys():
            self.print_arg_help()
        #from argv

        self.n_threads= arg_parser(argv,"n_threads",default_val=1)
        self.dummy_vocab_size = arg_parser(argv,"dummy_vocab_size",default_val=30000)

        self.quiet=arg_parser(argv,"quiet",default_val="False")
        self.debug_dir=arg_parser(argv,"debug_dir",default_val="./debug/")
        self.file = arg_parser(argv,"file",required=True)
        self.out_voc_file = arg_parser(argv,"voc",required=True)
        self.shrinking_rate = arg_parser(argv,"shrinking_rate",default_val=0.75)
        self.vocab_size = arg_parser(argv,"vocab_size",default_val=8000,required=True)
        self.seed_sentence_piece_size = arg_parser(argv,"seed_sentence_piece_size",default_val=1e5)
        self.use_original_make_seed = arg_parser(argv,"use_original_make_seed",default_val=False)
        self.unk_surface=arg_parser(argv,"unk_surface",default_val="⁇")
        self.character_coverage = arg_parser(argv,"character_coverage",default_val=1)
        # original spの"_"の太文字みたいな文字
        self.sep_voc = arg_parser(argv,"sep_voc",default_val=chr(9601))
        self.debug = arg_parser(argv,"debug",default_val=False)
        ###TODO
        self.debug=False
        # Merge all sentences into one array with 0x0000 delimiter
        self.kSentenceBoundary = arg_parser(argv,"kSentenceBoundary",default_val=chr(0x0000))


        self.debug_cnt=0
        self.SentencePiece = SentencePiece()
        self.Trie = None
        #self.sentences = []
        self.words = []
        self.desired_voc_size = int(self.vocab_size*1.1)
        self.required_chars=dict()

        if not self.quiet:
            print("argv")
            for key,val in argv.items():
                print("key:{} => {}".format(key,val))


    def print_arg_help(self):
        print("file: corpus path: required")
        print("voc: output vocablary file path: required")
        print("shirinking rate: shrinking rate in prune step: default 0.75")
        print("vocab_size: final vocabulary size : default 8000")
        print("seed_sentence_piece_size: default 1e5")
        print("use original_make_seed: if False, call spm_train and get seed piece")
        print("debug: if True, create pickle file")


    def read_sentencenpiece_from_voc_file(self, path:str):
        """
        trained vocからsentencepiceを読み取って、モデルにセットする
        Arguments:
            path(str): path to trained voc file
        """
        Voc = {}
        with open(path,encoding="utf-8") as f:
            for s in f:
                key, val = s.split("\t")
                Voc[key] = float(val)

        if self.debug:
            self.set_sentence_piece(Voc,debug_name="seed",info="init_piece")
        else:
            self.set_sentence_piece(Voc)

    def load_seed_sentencepiece_from_file(self)->dict:
        """c++のsentencepieceのmake_seedを呼び出してvocをとってくる。
        unigram_model_trainer.c++のEM前で、seed_pieceを求めた後に、fileに SaveVocab()と似た感じで、fileに書き込んで終了している。
        EMに入る前で止めている。
        """
        f_name=self.file.split("/")[-1]
        if os.path.isfile(f_name+".seed.vocab"):
            print("seed file is already exsists. skip c++ code")
        else:
            print("run MakeSeedSentence of original c++ sentnecepiece code to get initial piece")
            try:
                #TODO optionはこれでいいのか?
                #res = subprocess.run(["../../src/build_spm/src/spm_train","--input",self.file,"--model_prefix",f_name+".seed","--seed_sentencepiece_size",str(self.seed_sentence_piece_size),"--vocab_size",str(self.vocab_size)])
                #res = subprocess.run(["../../sentencepiece/build/src/spm_train","--input",self.file,"--model_prefix",f_name+".seed","--seed_sentencepiece_size",str(self.seed_sentence_piece_size),"--vocab_size",str(self.seed_sentence_piece_size)])
                res = subprocess.run(["../../../sentencepiece/build/src/spm_train","--input",self.file,"--model_prefix",f_name+".seed","--seed_sentencepiece_size",str(self.seed_sentence_piece_size),"--vocab_size",str(self.dummy_vocab_size)])
                #res = subprocess.run(["../src/build_spm/src/spm_train","--input",self.file,"--model_prefix",f_name+".seed","--seed_sentencepiece_size",str(self.seed_sentence_piece_size),"--character_coverage","1","--normalization_rule_name","identity","split_by_number","false"])
                #res = subprocess.run(["../../src/build_spm/src/spm_train","--input",self.file,"--model_prefix",f_name+".seed","--seed_sentencepiece_size",str(self.seed_sentence_piece_size),"--character_coverage","1","--normalization_rule_name","identity","split_by_number","false"])
            except:
                assert 1==2,"run error of spm_train"
                exit()

        Voc={}
        with open(f_name+".seed.vocab",encoding="utf-8") as f:
            for s in f:
                key,val = s.split("\t")
                Voc[key]=float(val)
        return Voc


    def build_trie(self):
        """ building Trie from piece
        """
        Trie = pygtrie.Trie()
        for (key, score) in self.SentencePiece.get_pieces().items():
            Trie[key] = (key, score)
        self.Trie = Trie

    def make_seed(self)->dict:
        """
        self.use_original_make_seed=Trueならoriginal spm_trainで作る
        else: myimpoleで作る(myimpleを使う意味はない
        """
        if self.use_original_make_seed:
            seed_sentencepieces = self.load_seed_sentencepiece_from_file()
        else:
            assert "not ok"
        return seed_sentencepieces

    def load_sentence(self,path=None):
        """ load sentence from file
        引数のpathはdecodeとかencodeの時に使う
        """

        #print("load_ sentence")
        if path is None:
            path = self.file

        sentences = []
        self.words = defaultdict(int)
        chars=defaultdict(int)
        with open(path,encoding="utf-8") as f:
            for s in f:
                #originalは半角のみを扱っていたので、半角のみを扱うようにする。
                # _s = "_"+"_".join(s.split(" "))#全角と半角のspaceを区別するか(\tとか\nもsplitされるs.split())
                s = s.replace("\n","")#\nを消す感じ
                _s = self.sep_voc + self.sep_voc.join(s.split(" "))

                sentences.append(_s)
                for w in s.split(" "):
                    self.words[self.sep_voc+w] += 1
                    for c in w:
                        if c=="\t":
                            continue
                        chars[c]+=1

        #determines uequired_chars which must be included in the vocabulary
        accumulated_chars_count=0
        all_chars_count = sum(chars.values())
        for key,val in sorted(chars.items(),key=lambda x:-x[1]):
            coverage = accumulated_chars_count/all_chars_count
            if coverage>=self.character_coverage:
                break
            accumulated_chars_count+=val
            assert key!=chr(0x0020),"space must not be included"
            assert key!="\t","tab must not be included"
            self.required_chars[key]=val

        if self.debug:
            print("Alphabet size=>",len(self.required_chars))
            print("Final character cpverage=>",accumulated_chars_count/all_chars_count)
                
        assert self.character_coverage==1,"unk 処理 is not implemented at load sentences #TODO"
        assert len(self.required_chars)<=self.vocab_size,"vocab_size is too small, should larger than required_chars_size:{}".format(len(self.required_chars))


        return sentences


    def set_sentence_piece(self, pieces,debug_name=None,info=None):
        """ set piece into Sentencepiece class
        Always call build_trie to create new Trie corresponding to new_pieces
        Args:
            pieces(dict): current sentencepieces dict[piece]=score
        """

        self.SentencePiece._set_sentence_piece(pieces)
        self.build_trie()

    def run_e_step_pool(self):

        """E step of EM learning
        Return:
            objective(int): int
            nun_token(int): sum of the token num of Viterbi path
            expected(dict): dict[piece]=score of the piece
        """
        expected = defaultdict(int)
        objective = 0
        num_tokens = 0

        all_sentence_freq = sum(self.words.values())

        #words.items()をthread数のlistに分割する。
        size=len(self.words.items())//self.n_threads+1
        iterable=[(items,self.SentencePiece.get_pieces(),self.Trie) for items in  zip_longest(*[iter(self.words.items())]*size)]

        with Pool(processes=self.n_threads) as p:
            ret=p.map(func=process_each_estep, iterable=iterable)

        for ret_exp, ret_obj, ret_tokens in ret:
            for key, val in ret_exp.items():
                expected[key]+=val
            num_tokens+=ret_tokens
            objective+=ret_obj


        objective/=all_sentence_freq #orginal実装では、割ってから足している。並列化のため、足してから割る。

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
                print("skip zero freq")

            if freq < kExpectedFrequencyThreshold:
                #print("remove m key:{} score:{}".format(key,expected[key]))
                continue
            new_pieces[key] = freq
            sum_freq += freq
        print("M step filtered infrequent sentencepiece, {} pieces removed".format(
            self.SentencePiece.get_piece_size()-len(new_pieces)))

        logsum = Digamma(sum_freq)
        for key, val in new_pieces.items():
            new_pieces[key] = Digamma(val)-logsum

        print("m_step:sum(p(x))=>{}".format(sum([exp(v) for v in new_pieces.values()])))

        return new_pieces

    def run_EM(self):
        for itr in range(2):  # EM iteration loop

            start = time.time()
            expected, objective, num_tokens = self.run_e_step_pool()
            print("Estep=>",time.time()-start)
            #expected, objective, num_tokens = self.run_e_step()

            new_sentencepieces = self.run_m_step(expected)

            self.set_sentence_piece(new_sentencepieces)

            piece_size = self.SentencePiece.get_piece_size()
            print("EM sub_iter= {} size={} obj={} num_tokens= {} num_tokens/piece= {}".format(
                itr, piece_size, objective, num_tokens, num_tokens/piece_size))


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

        #print("alt=>",alternatives)
        return always_keep, alternatives

    def prune_step_2_freq_inverted_pool(self):
        """
        Return
            vsum(float):
            freq(dict):
            inverted(dict):
        """
        vsum = 0
        freq = defaultdict(int)
        # inverted[key] stires the set of sentence index where the sentencepiece (key) appears
        inverted = defaultdict(int)


        size=len(self.words.items())//self.n_threads+1
        iterable=[(items,self.SentencePiece.get_pieces(),self.Trie) for items in  zip_longest(*[iter(self.words.items())]*size)]

        with Pool(processes=self.n_threads) as p:
            ret = p.map(func=process_each_prune, iterable=iterable)
            
            # remove this
        for ret_vsum,ret_freq,ret_inverted in ret:
            vsum+=ret_vsum
            for key,val in ret_freq.items():
                freq[key]+=val
            for key,val in ret_inverted.items():
                inverted[key]+=val

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
                F= freq[key]
                assert inverted[key]==freq[key]
                F /= vsum  # keyが出てくる文の数を全文数で割ったもの
                # keyの出現確率( P(x)= \frac{freq_x}{sum(all_piece_freq)})
                logprob_sp = log(freq[key])-logsum
                # x->x_altに置換後の log(freq_sum)
                logsum_alt = log(sum_freq+freq[key]*(self.SentencePiece.get_piece_size()-1))

                logprob_alt = 0
                for alt in alternatives[key]:
                    logprob_alt += log(freq[alt]+freq[key])-logsum_alt

                # Freq*(logp(x)-logp(x_alts))
                #(P(X)よりP(x_alt)の方が高いとき、logp(x)= -大 logp(x_alt)=-小->loss=-大
                #P(x)が小さい場合->pieceを分けた方がいい。
                loss = F*(logprob_sp-logprob_alt)
                candidate[key]=loss


        return candidate, new_sentencepieces

    def prune_step_4_prune_candidate(self, candidate, new_sentencepieces):
        """
        Return
            candidate(dict): dict[key] = loss of key
            new_sentencepieces(dict):
        """
        assert  len(new_sentencepieces)<=self.desired_voc_size,"{}".format(len(new_sentencepieces))

        current_piece = self.SentencePiece.get_pieces()
        pruned_size =\
                max(int(len(current_piece)*self.shrinking_rate), self.desired_voc_size)

        candidate_list = [(key, val) for key, val in candidate.items()]
        #for piece, _ in sorted(candidate_list, key=lambda x: x[1], reverse=True):
        #lossはp(x)<p(x_alt)の時に、-大(xを分割したいとき),逆に loss=が大きい時は、p(x)を残した方がいいとき。よって、sorted(reverse)

        for piece, score in sorted(candidate_list, key=lambda x: x[1], reverse=True):
            # add piece from candidate in decsengind order of score till piece size reaches to pruned_size
            #assert len(new_sentencepieces)>=pruned_size,"remove this code"
            if len(new_sentencepieces) == pruned_size:
                break
            new_sentencepieces[piece] = current_piece[piece]
        #assert len(current_piece)==len(new_sentencepieces),"no piece is  pruned"

        return new_sentencepieces

    def prune_piece(self):
        # First,
        always_keep, alternatives = self.prune_step_1_always_keep_alternative()
        # Second, segments all sentences to compute likelihoood with a Unigram LM
        #vsum, freq, inverted = self.prune_step_2_freq_inverted()

        vsum, freq, inverted = self.prune_step_2_freq_inverted_pool()
        # Third
        candidate, new_sentencepieces = self.prune_step_3_new_piece_cand(
            always_keep, alternatives, vsum, freq, inverted)
        # Forth,
        new_sentencepieces = self.prune_step_4_prune_candidate(
            candidate, new_sentencepieces)

        assert self.SentencePiece.get_piece_size()!=len(new_sentencepieces),"no piece is  pruned"
        print("pruned {} pieces".format(self.SentencePiece.get_piece_size()-len(new_sentencepieces)))
        return new_sentencepieces

    def check_finish(self):
        return self.SentencePiece.get_piece_size() <= self.desired_voc_size

    def finalize_sentencepiece(self):
        """最終的な処理
        fileへの書き込みをする
        "vocab_size*1.1のpieceが入ってくるので、vocabsizeまで削るはず
        """

        ##確定処理
        final_piece = dict()

        min_score_penalty = 0.0
        kMinScorePenaltyDelta = 0.0001

        pieces = self.SentencePiece.get_pieces()
        for key,val in sorted(self.required_chars.items(),key=lambda x:-x[1]):
            if key in pieces.keys():
                final_piece[key]=pieces[key]
            else:
                final_piece[key] = min(pieces.values())+min_score_penalty
                min_score_penalty+=kMinScorePenaltyDelta

        for  key,val in sorted(pieces.items(),key=lambda x:-x[1]):
            if key in final_piece.keys():
                continue
            if len(final_piece)==self.vocab_size-3:
                break
            final_piece[key]=val


        self.set_sentence_piece(final_piece,debug_name="finalized_piece")
        #required charをどっかで作っているはず#trainer_interface.cc の LoadSentenceの下の方でやってる
        #write out to file
        piece = self.SentencePiece.get_pieces()
        with open(self.out_voc_file, "w") as f:
            for key in ["<unk>","<s>","</s>"]:
                f.write("{}\t{}\n".format(key,0))
            for key, val in sorted(piece.items(), key=lambda x: -x[1]):
                f.write("{}\t{}\n".format(key, val))
        print("finalized vocab size=>",len(piece))
        print("written voc to {}".format(self.out_voc_file))

    def train(self):
        """ training 
        """
        self.load_sentence()
        seed_sentencepieces = self.make_seed()
        self.set_sentence_piece(seed_sentencepieces)
        #self.set_sentence_piece(seed_sentencepieces,debug_name="train_start")

        step_cnt = 0
        print("init vocab size is {}\n start EM trainind".format(self.SentencePiece.get_piece_size()))

        #TODO RUN_EMで一つにまとめる
        while True:

            step_cnt += 1
            self.run_EM()
            if self.check_finish():
                break
            new_sentencepieces = self.prune_piece()
            if self.debug:
                self.set_sentence_piece(new_sentencepieces,debug_name="step{}_prune".format(step_cnt))
            else:
                self.set_sentence_piece(new_sentencepieces)
        # Save to file
        print("{} step is needed to converge".format(step_cnt))
        self.finalize_sentencepiece()

    def encode_one_sent(self, sent):
        #TODO encode_poolがうまくいくなら決して良い
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

    def encode(self, corpus):
        """
        self.sentencesを全てencode_one()して、listにしてreturn?
        corpus: path to  corpus

        Returns:
            encode_sentences(list):
        """
        sentences = self.load_sentence(path=corpus)
        size = len(sentences)//self.n_threads+1

        iterable = [(items, self.SentencePiece.get_pieces(), self.Trie) for items in zip_longest(*[iter(sentences)]*size)]

        with Pool(processes=self.n_threads) as p:
            ret=p.map(func=process_each_encode, iterable=iterable)

        encode_sent=[]
        for ss in ret:
            for s in ss:
                encode_sent.append(s)

        assert all("".join(a.split(" "))==b for a,b in zip(encode_sent[:10], sentences[:10]))
        assert len(sentences)==len(encode_sent)
        return encode_sent


    def decode_one_piece(self,piece:str):
        #TODO 消す
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

    def load_voc_file(self,voc_file):
        vocs=set()
        with open(voc_file,encoding="utf-8") as f:
            for s in f:
                key,val  = s.split()
                vocs.add(key)
        self.decode_voc=vocs


    def decode_one_sent(self, s:str)->str:
        """
        Arguments:
            tokenized_sent(str): tokenizeされた文
        Returns:
            ret_sent(str): space split tokenize sentence
        """
        s = "".join([c if c in self.decode_voc else self.unk_surface  for c in s.split()])
        s = "".join(s.split(" ")) #remove whitespace
        s = s.replace("\n","")
        s = s.replace(self.sep_voc," ")

        if len(s)>0 and s[0]==" ": #remove white space located at the beginning of sent(beggining "_")
            s=s[1:]
        return s


    def decode(self,path:str)->list:
        """
        Arguments: decodeしたい文のpath

        """
        #軽そうだし、parallelにしなくてもいいか。
        ret=[]

        with open(path,encoding="utf-8") as f:
            for s in f:
                ret.append(self.decode_one_sent(s))
        return ret



def process_each_estep(tup):

    expected = defaultdict(int)
    objective = 0
    num_tokens = 0

    (items,pieces,trie)=tup
    L = Lattice()
    for item in items:
        if item is None:
            continue
        (key,freq)=item
        L.set_sentence(key)
        L.populate_nodes(pieces,trie)
        Z, ret_expected = L.populate_marginal(freq)

        for key, val in ret_expected.items():
            expected[key] += val

        N = len(L.Viterbi())
        num_tokens += N
        objective-= Z
    return (expected, objective, num_tokens)

def process_each_prune(tup):
    """
    poolで呼ばれる関数。
    classのなかにかくと、classごとcopyされてeach processに渡される。
    それを防ぐために、外に書く

    1文ずつの処理。(sent,piece,trie)よりも、まとめた方が早い(sent_list, piece,trie)
    """
    (items,piece,trie) = tup

    vsum = 0
    freq = defaultdict(int)
    inverted = defaultdict(int)

    L = Lattice()
    for item in items:
        if item is None:
            continue

        (s,score) = item
        vsum += score
        L.set_sentence(s)
        L.populate_nodes(piece, trie)

        for word in L.Viterbi(ret_piece=True):
            freq[word] += score
            inverted[word] += score
    return (vsum,freq,inverted)

def process_each_encode(tup):
    """
    tup: tuple(sentence_list, piece, trie)
    
    return: tokenized_sentence_list
    """
    (items,piece,trie)=tup


    ret=[]
    L = Lattice()
    for sent in items:
        if sent is None:
            continue
        L.set_sentence(sent)
        L.populate_nodes(piece, trie)
        tokenize_sent = " ".join(L.Viterbi(ret_piece=True))
        ret.append(tokenize_sent)
        assert "".join(tokenize_sent.split(" "))==sent
    return ret
