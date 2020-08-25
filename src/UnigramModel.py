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


class UnigramModel:
    """
    """

    def __init__(self, argv):
        """ get parameter from argv
        """
        if "help" in argv.keys():
            self.print_arg_help()
        #from argv
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
        # Merge all sentences into one array with 0x0000 delimiter
        self.kSentenceBoundary = arg_parser(argv,"kSentenceBoundary",default_val=chr(0x0000))


        self.debug_cnt=0
        self.SentencePiece = SentencePiece()
        self.Trie = None
        self.sentences = []
        self.words = []
        self.desired_voc_size = int(self.vocab_size*1.1)
        self.required_chars=dict()

        if not self.quiet:
            print("argv")
            for key,val in argv.items():
                print("key:{} => {}".format(key,val))
        print("desired_voc_size=>",self.desired_voc_size)


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
        with open(path) as f:
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
        print("fname=>",f_name)
        if os.path.isfile(f_name+".seed.vocab"):
            print("seed file is already exsists. skip c++ code")
        else:
            print("run MakeSeedSentence of original c++ sentnecepiece code to get initial piece")
            try:
                #TODO optionはこれでいいのか?
                res = subprocess.run(["../../src/build_spm/src/spm_train","--input",self.file,"--model_prefix",f_name+".seed","--seed_sentencepiece_size",str(self.seed_sentence_piece_size)])
                #res = subprocess.run(["../src/build_spm/src/spm_train","--input",self.file,"--model_prefix",f_name+".seed","--seed_sentencepiece_size",str(self.seed_sentence_piece_size),"--character_coverage","1","--normalization_rule_name","identity","split_by_number","false"])
                #res = subprocess.run(["../../src/build_spm/src/spm_train","--input",self.file,"--model_prefix",f_name+".seed","--seed_sentencepiece_size",str(self.seed_sentence_piece_size),"--character_coverage","1","--normalization_rule_name","identity","split_by_number","false"])
            except:
                assert 1==2,"run error of spm_train"
                exit()

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


        for (word, freq) in self.words.items():
            # ここでpretolenizeってのをかましている
            for c in word:
                assert c == UnicodeCharToUTF8(UTF8ToUnicodeText(c))
                assert isValidCodepoint(c),"isValidCodepoint {}".format(c)
                assert c!=0x0020,"space must not be included"
                if c== self.kSentenceBoundary:
                    print("Find null char")
                    continue
                array.append(c)
                all_chars[c] += freq
            array.append(self.kSentenceBoundary)

        if not self.quiet:
            print("alphabet=>", len(all_chars))

        # make a suffix_array to extract all sub strings occuring more than 2 times in the sentence
        # ここは、配列のサイズによって、分割した方が良さそうな気がする?sa-isでやっているはずで、線形アルゴリズムだから関係ない説もある
        A = "".join(array)
        if not self.quiet:
            print("Making Suffix Array len:{}".format(len(array)))
        SA = SuffixArray(A)

        if not self.quiet:
            print("Extracting frequent sub strings...")
        # TODO 結構怪しい気がする ここの処理
        substr = set()
        for i, l in enumerate(SA.longest_common_prefix()):
            if l <= 1:  # lcp=1なので1回しか出てこない
                continue
            sb = SA.string[SA.sa[i]:SA.sa[i]+l]  # 2回以上出てくるsbst
            if sb[-1] == self.kSentenceBoundary:  # 最後の "0x00"は大目に見る
                sb = sb[:-1]
            if len(sb) <= 1:  # 多目に見た後に長さが2.elseはsb=charになっている
                continue
            if any(v == self.kSentenceBoundary for v in sb):  # 途中に 0x00が入っているのはinvalid
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
            if not self.quiet:
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

        if not self.quiet:
            print("Initialized {} seed sentence pieces".format(
            len(seed_sentencepieces)))
        return seed_sentencepieces

    def dump_to_pickle(self,name,data):
        """
        dump data into pickle
        """
        with open(self.debug_dir+"{:3}_".format(self.debug_cnt)+name+".pickle","wb") as f:
                pickle.dump(data,f)


    def set_sentence_piece(self, pieces,debug_name=None,info=None):
        """ set piece into Sentencepiece class
        Always call build_trie to create new Trie corresponding to new_pieces
        Args:
            pieces(dict): current sentencepieces dict[piece]=score
        """
        if self.debug and self.SentencePiece.get_pieces() is not None:
            ##LM OBJを求める
            _, obj_before,_ = self.run_e_step()
            pruned_voc = set(self.SentencePiece.get_pieces().keys() - pieces.keys())
            #pieceの更新
            self.SentencePiece._set_sentence_piece(pieces)
            self.build_trie(pieces)
            ###
            _, obj_after,_ = self.run_e_step()

            debug_info={"obj_before":obj_before,"obj_after":obj_after,"pruned_voc":pruned_voc,"info":info,"gain":obj_before-obj_after}
            self.dump_to_pickle(debug_name,debug_info)
            self.debug_cnt+=1

        self.SentencePiece._set_sentence_piece(pieces)
        self.build_trie(pieces)

    def load_sentence(self,path=None):
        """ load sentence from file
        引数のpathはdecodeとかencodeの時に使う
        """
        if path is None:
            path = self.file

        sentences = []
        words = defaultdict(int)
        chars=defaultdict(int)
        with open(path) as f:
            for s in f:
                #originalは半角のみを扱っていたので、半角のみを扱うようにする。
                # _s = "_"+"_".join(s.split(" "))#全角と半角のspaceを区別するか(\tとか\nもsplitされるs.split())
                s = s.replace("\n","")#\nを消す感じ
                _s = self.sep_voc + self.sep_voc.join(s.split(" "))
                for w in s.split(" "):
                    words[self.sep_voc+w] += 1
                    for c in w:
                        if c=="\t":
                            continue
                        chars[c]+=1
                sentences.append(_s)

        self.sentences = sentences
        self.words = words

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


    def run_e_step(self):
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
        vsum, freq, inverted = self.prune_step_2_freq_inverted()
        # Third
        candidate, new_sentencepieces = self.prune_step_3_new_piece_cand(
            always_keep, alternatives, vsum, freq, inverted)
        # Forth,
        new_sentencepieces = self.prune_step_4_prune_candidate(
            candidate, new_sentencepieces)

        assert self.SentencePiece.get_piece_size()!=len(new_sentencepieces),"no piece is  pruned"
        print("pruned {} pieces".format(self.SentencePiece.get_piece_size()-len(new_sentencepieces)))
        return new_sentencepieces

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

    def build_trie(self, pieces):
        """ building Trie from piece
        """
        Trie = pygtrie.Trie()
        for (key, score) in pieces.items():
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
            seed_sentencepieces = self.make_seed_sentence_piece()
        return seed_sentencepieces

    def check_finish(self):
        return self.SentencePiece.get_piece_size() <= self.desired_voc_size

    def run_EM(self):
        for itr in range(2):  # EM iteration loop
            expected, objective, num_tokens = self.run_e_step()
            new_sentencepieces = self.run_m_step(expected)

            self.set_sentence_piece(new_sentencepieces)

            piece_size = self.SentencePiece.get_piece_size()
            print("EM sub_iter= {} size={} obj={} num_tokens= {} num_tokens/piece= {}".format(
                itr, piece_size, objective, num_tokens, num_tokens/piece_size))


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
            #for itr in range(2):  # EM iteration loop
            #    expected, objective, num_tokens = self.run_e_step()
            #    new_sentencepieces = self.run_m_step(expected)

            #    if self.debug:
            #        self.set_sentence_piece(new_sentencepieces,debug_name="step{}_mstep{}".format(step_cnt,itr))
            #    else:
            #        self.set_sentence_piece(new_sentencepieces)

            #    piece_size = self.SentencePiece.get_piece_size()
            #    print("EM sub_iter= {} size={} obj={} num_tokens= {} num_tokens/piece= {}".format(
            #        itr, piece_size, objective, num_tokens, num_tokens/piece_size))

            ##TODO 外から呼べるように bool func(){}にする
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

    def load_voc_file(self,voc_file):
        vocs=set()
        with open(voc_file) as f:
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
