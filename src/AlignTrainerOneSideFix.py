import sys
sys.path.append("./translate")

from UnigramModel import UnigramModel
from Lattice import Lattice

from collections import defaultdict
from math import log,exp
from itertools import zip_longest
from multiprocessing import Pool

import pickle
import random
import time

from translate import IBMModel1, IBMModel, Alignment, AlignedSent


class AlignTrainerOneSideFix:
    """
        両言語のうち、片方を固定して学習する。
        prune stepにalignment lossを加える
    """

    def __init__(self,arg_src, fix_side_corpus):
        """
            arg_src: UnigramModelのparam
            fix_side_corpus: すでにtokenizeした固定する方の言語のcorpus
        """

        self.U_src = UnigramModel(arg_src)

        self.n_threads=self.U_src.n_threads
        self.src_tokenised=[]
        self.tgt_tokenised=[]

        self.tgt_tokenize_corpus=[]
        print("init")

        with open(fix_side_corpus) as f:
            for s in f:
                self.tgt_tokenize_corpus.append(s.split())

    def prepare_UnigramModel(self):
        # load sentence
        print("load_sentence")
        self.src_sentences=self.U_src.load_sentence()


        # seed_piece
        print("make seed")
        seed_src = self.U_src.make_seed()

        self.U_src.set_sentence_piece(seed_src)


    def align_src_func(self, always_keep, alternatives, freq):
        return self.alignment_loss(always_keep, alternatives, freq,src_turn=True)

    def prune_step_with_align(self):
        """
        train
        :param: pruneLoss=(1-alpha)*LMloss+(alpha)*AlignLoss
        """

        always_keep_s, alternatives_s = self.U_src.prune_step_1_always_keep_alternative()

        vsum_s, freq_s, inverted_s = self.U_src.prune_step_2_freq_inverted_pool()

        LM_loss_s, new_sentencepieces_s = self.U_src.prune_step_3_new_piece_cand(
            always_keep_s, alternatives_s, vsum_s, freq_s, inverted_s)

        start = time.time()
        self.tokenize_viterbi_pool()
        print("time To tokenize=>",time.time()-start)

        align_loss_s = self.align_src_func(always_keep_s,alternatives_s,freq_s)

        joint_loss_s = dict()
        for key in LM_loss_s.keys():
            joint_loss_s[key] = (1-self.alpha)*LM_loss_s[key]+self.alpha*align_loss_s[key]

        new_piece_s =self. U_src.prune_step_4_prune_candidate(
            joint_loss_s, new_sentencepieces_s)

        assert not(self.U_src.SentencePiece.get_piece_size()==len(new_piece_s) ),"no piece is  pruned"

        return new_piece_s

    def train(self,alpha=0.01,sample_rate=1.0, em_steps=5,back_up_interval=-1,back_up_file=None):
        """
        alpha: pruneScore = LMscore*(1-alpha)+ PruneScore*alpha
        sample_rate: when train ibm1Model, smaple from whole corpus with this  rate
        em_steps: em_steps of ibmModel
        """

        self.alpha=alpha
        self.sample_rate=sample_rate
        self.em_steps=em_steps
        self.back_up_interval=back_up_interval
        self.back_up_file=back_up_file

        assert self.back_up_interval==-1 or self.back_up_file is not None, "set backup path"
        print("Train align")
        self.prepare_UnigramModel()
        # Start EM
        print("Seed voc size=> src:{}\nStart EM training".format(self.U_src.SentencePiece.get_piece_size()))

        step_cnt = 0
        while True:
            step_cnt += 1
            print("EM_src")
            self.U_src.run_EM()
            if self.U_src.check_finish():
                break

            print("Align")
            new_piece_src = self.prune_step_with_align()

            self.U_src.set_sentence_piece(new_piece_src)

            if self.back_up_interval>=1 and step_cnt%self.back_up_interval==0:
                print("back up")
                with open(back_up_file+"_{}.src.pickle".format(step_cnt),"wb") as f:
                    pickle.dump(self.U_src.SentencePiece.get_pieces(),f)


        print("{} step is needed to converge".format(step_cnt))
        self.U_src.finalize_sentencepiece()

    def alignment_loss(self,always_keep_s, alternatives_s, freq_s,src_turn=True):
        """ alignlossを求めたい
        U_sにalignment lossを加える
        * X,Y,A全てbestを使って近似

        Arguments:
            U_s(class obj): source UnigramMode
            U_t(class obj): target UnigramModel
            always_keep_s(dict):dict[key]=bool whether keep the piece always or  not
            alternatives_s(dict): dict[key]=[list]. dict[piece]=sequence of its alternatives
            freq_s(dict): occurence num of the word in viterbi path on whole corpus

        Memo
        * ibm1.translation_table[tgt][src]は、sum(tt[tgt].values())!=1で、sum(tt[tgt].values)-tt[tgt][None]だと大体1になる。(浮動小数点のごさ)
        * sum(tt[t][src] for t in tt.keys())=1 tgtはNoneを含まないから
        """

        print("get_bitexts src=>",src_turn)

        # Train IBM Model1 with best tokenize sentence of source and target(bitext,iteration)
        bitexts=[]
        for src,tgt in zip(self.src_tokenised,self.tgt_tokenised):
            bitexts.append(AlignedSent(tgt,src))

        print("train ibm {}steps".format(self.em_steps))
        start = time.time()
        ibm1 = IBMModel1(bitexts, self.em_steps)
        print("finish train ibm->",time.time()-start)


        # for each piece x,get words which aligns to x
        #AlignedWords[key1][key2]=val, key1にalignするkey2の数
        AlignedWords = defaultdict(lambda: defaultdict(int))
        AlignedCnt = defaultdict(int)

        #alignは、word->mot(tgt->src)
        for bitext in bitexts:
            # align=(idx_in_tgt,idx_in_src)
            tgt, src, align = bitext.words, bitext.mots, bitext.alignment
            for (idx_tgt, idx_src) in align:
                if idx_src is None:
                    AlignedCnt["None"] += 1
                    continue  # したのalignedwordを使うところで、Noneは使わないから、countしなくてよさそう
                AlignedWords[src[idx_src]][tgt[idx_tgt]] += 1
                AlignedCnt[src[idx_src]] += 1

        candidate_s = dict()
        all_align_cnt = 0
        no_align_cnt = 0


        items=self.U_src.SentencePiece.get_pieces().items()
        for s_key, _ in items:
            if freq_s[s_key] == 0 or not always_keep_s[s_key]:
                continue
            elif len(alternatives_s[s_key]) == 0:
                continue
            else:
                loss = 0
                # translation_table[t][s]=P(t|s),tgt tがsrc sにalignする確率
                all_align_cnt += 1
                if len(AlignedWords[s_key].items()) == 0:
                    no_align_cnt += 1

                sum_val = sum(AlignedWords[s_key].values())
                for t_key, val in AlignedWords[s_key].items():
                    p_t_s = ibm1.translation_table[t_key][s_key]
                    #altのなかで、一番 alignの付け替えられそうなものに付け替える
                    p_t_s_alt = max(
                        ibm1.translation_table[t_key][s_key_alt] for s_key_alt in alternatives_s[s_key])

                    logP_key = log(p_t_s)  # logP(t|x)
                    # P(t|x)がx_altにequally distributed
                    logP_alt = log(p_t_s_alt+p_t_s/len(alternatives_s[s_key]))

                    loss += val/sum_val*(logP_key - logP_alt)
                candidate_s[s_key] = loss
        print("end calc of align_loss")
        return candidate_s

    def tokenize_viterbi_pool(self):

        self.src_tokenised=[]
        self.tgt_tokenised=[]

        len_examples=len(self.src_sentences)
        use_examples= int(len_examples*self.sample_rate)
        use_idx=set(random.sample(range(len_examples),use_examples))


        print("all:{} use:{} sample_rate:{}".format(len_examples, use_examples,self.sample_rate))
        print("len(set)=>",len(use_idx))


        size=use_examples//self.n_threads +1
        use_src =[s for i,s in enumerate(self.src_sentences) if i in use_idx]
        iterable=[(items,self.U_src.SentencePiece.get_pieces(),self.U_src.Trie) for items in  zip_longest(*[iter(use_src)]*size)]

        print("src_viterbi")
        with Pool(processes=self.n_threads) as p:
            src_viterbis=p.map(func=process_each,iterable=iterable)

        tgt_viterbis=[s for i,s in enumerate(self.tgt_tokenised) if i in use_idx]


        assert len(self.src_tokenised)==len(self.tgt_tokenised)
        print("create bitexts")
        for ret_src, ret_tgt in zip(src_viterbis,tgt_viterbis):
            for src,tgt in zip(ret_src,ret_tgt):
                self.src_tokenised.append(src)
                self.tgt_tokenised.append(tgt)



def process_each(tup):
    (items,piece,trie) = tup

    L = Lattice()

    ret=[]
    for item in items:
        if item is None:
            continue

        L.set_sentence(item)
        L.populate_nodes(piece, trie)
        ret.append(L.Viterbi(ret_piece=True))

    return ret
