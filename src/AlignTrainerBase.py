# lattice とUnigramをimport するために必要
import sys
from math import log,exp
from Lattice import Lattice
from UnigramModel import UnigramModel
from collections import defaultdict
import pickle
import random
sys.path.append("./translate")
from translate import IBMModel1
from translate import IBMModel
from translate import Alignment
from translate import AlignedSent
from Lattice import Lattice

from multiprocessing import Pool

class AlignTrainerBase:
    def __init__(self,arg_src, arg_tgt):

        self.U_src = UnigramModel(arg_src)
        self.U_tgt = UnigramModel(arg_tgt)
        print("init")

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


    #@abstractmethod
    def align_src_func(self,always_keep,alternatives,freq):
        pass
    #@abstractmethod
    def align_tgt_func(self,always_keep,alternatives,freq):
        pass


    def prune_step_with_align(self ,alpha=0.01):
        """
        train
        :param: pruneLoss=(1-alpha)*LMloss+(alpha)*AlignLoss
        """

        always_keep_s, alternatives_s = self.U_src.prune_step_1_always_keep_alternative()
        always_keep_t, alternatives_t = self.U_tgt.prune_step_1_always_keep_alternative()

        vsum_s, freq_s, inverted_s = self.U_src.prune_step_2_freq_inverted()
        vsum_t, freq_t, inverted_t = self.U_tgt.prune_step_2_freq_inverted()

        LM_loss_s, new_sentencepieces_s = self.U_src.prune_step_3_new_piece_cand(
            always_keep_s, alternatives_s, vsum_s, freq_s, inverted_s)
        LM_loss_t, new_sentencepieces_t = self.U_tgt.prune_step_3_new_piece_cand(
            always_keep_t, alternatives_t, vsum_t, freq_t, inverted_t)

        align_loss_s = self.align_src_func(always_keep_s,alternatives_s,freq_s)
        align_loss_t = self.align_tgt_func(always_keep_t,alternatives_t,freq_t)

        joint_loss_s = dict()
        joint_loss_t = dict()
        for key in LM_loss_s.keys():
            joint_loss_s[key] = (1-alpha)*LM_loss_s[key]+alpha*align_loss_s[key]
        for key in LM_loss_t.keys():
            joint_loss_t[key] = (1-alpha)*LM_loss_t[key]+alpha*align_loss_t[key]

        new_piece_s =self. U_src.prune_step_4_prune_candidate(
            joint_loss_s, new_sentencepieces_s)
        new_piece_t = self.U_tgt.prune_step_4_prune_candidate(
            joint_loss_t, new_sentencepieces_t)

        assert not(self.U_src.SentencePiece.get_piece_size()==len(new_piece_s) and self.U_tgt.SentencePiece.get_piece_size()==len(new_piece_t)),"no piece is  pruned"

        return new_piece_s, new_piece_t

    def train(self,alpha=0.01,back_up_interval=-1,back_up_file=None):

        assert back_up_interval==-1 or back_up_file is not None, "set backup path"
        print("Train align")
        self.prepare_UnigramModel()
        # Start EM
        print("Seed voc size=> src:{} tgt:{}\nStart EM training".format(self.U_src.SentencePiece.get_piece_size(),self.U_tgt.SentencePiece.get_piece_size()))
        step_cnt = 0
        while True:
            step_cnt += 1
            print("EM_src")
            self.U_src.run_EM()
            print("EM_tgt")
            self.U_tgt.run_EM()
            if self.U_src.check_finish() and self.U_tgt.check_finish():
                break

            print("Align")
            new_piece_src, new_piece_tgt= self.prune_step_with_align(alpha=alpha)

            self.U_src.set_sentence_piece(new_piece_src)
            self.U_tgt.set_sentence_piece(new_piece_tgt)

            if back_up_interval>=1 and step_cnt%back_up_interval==0:
                with open(back_up_file+"_{}.src.pickle".format(step_cnt),"wb") as f:
                    pickle.dump(self.U_src,f)
                with open(back_up_file+"_{}.tgt.pickle".format(step_cnt),"wb") as f:
                    pickle.dump(self.U_tgt,f)


        print("{} step is needed to converge".format(step_cnt))
        self.U_src.finalize_sentencepiece()
        self.U_tgt.finalize_sentencepiece()

    def alignment_loss(self,always_keep_s, alternatives_s, freq_s,sample_rate=1.0,em_steps=2,src_turn=True):
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
        bitexts = self.get_bitexts(sample_rate=sample_rate,src_turn=src_turn)
        # Train IBM Model1 with best tokenize sentence of source and target(bitext,iteration)
        print("train ibm {}steps",em_steps)
        ibm1 = IBMModel1(bitexts, em_steps)
        print("finish train ibm")


        # for each piece x,get words which aligns to x
        #AlignedWords[key1][key2]=val, key1にalignするkey2の数
        AlignedWords = defaultdict(lambda: defaultdict(int))
        AlignedCnt = defaultdict(int)

        for bitext in bitexts:
            # align=(idx_in_tgt,idx_in_src)
            tgt, src, align = bitext.words, bitext.mots, bitext.alignment
            for (idx_tgt, idx_src) in align:
                if idx_src is None:
                    AlignedCnt["None"] += 1
                    continue  # したのalignedwordを使うところで、Noneは使わないから、countしなくてよさそう
                #print("src:{}, tgt:{}".format(src[idx_src],tgt[idx_tgt]))
                AlignedWords[src[idx_src]][tgt[idx_tgt]] += 1
                AlignedCnt[src[idx_src]] += 1

        candidate_s = dict()
        all_align_cnt = 0
        no_align_cnt = 0


        if src_turn:
            items=self.U_src.SentencePiece.get_pieces().items()
        else:
            print("tgt")
            items=self.U_tgt.SentencePiece.get_pieces().items()

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
                    p_t_s_alt = max(
                        ibm1.translation_table[t_key][s_key_alt] for s_key_alt in alternatives_s[s_key])

                    p_alt = p_t_s_alt+p_t_s/len(alternatives_s[s_key])
                    logP_key = log(p_t_s)  # logP(t|x)
                    # P(t|x)がx_altにequally distributed
                    logP_alt = log(p_t_s_alt+p_t_s/len(alternatives_s[s_key]))

                    loss += val/sum_val*(logP_key - logP_alt)
                candidate_s[s_key] = loss
        print("end calc of align_loss")
        return candidate_s

    def get_bitexts(self,sample_rate=1.0,src_turn=True):
        """
        srcとtgtをbest tokenizeしてreturn　する
        Arguments:
            U_s,U_t: Unigram model for source and tgt respectively
        Return
            bitexts(list): text pair for train ibm
        """
        bitexts = []


        len_examples=len(self.U_src.sentences)
        use_examples= int(len_examples*sample_rate)
        use_idx=set(random.sample(range(len_examples),use_examples))


        print("all:{} use:{} sample_rate:{}".format(len_examples, use_examples,sample_rate))
        for i,(src, tgt) in enumerate(zip(self.U_src.sentences, self.U_tgt.sentences)):
            if i not in use_idx:
                continue
            #src_viterbi = get_viterbi_path(src, U_s)
            #tgt_viterbi = get_viterbi_path(tgt, U_t)
            src_viterbi= self.U_src.get_viterbi_path(src)
            tgt_viterbi = self.U_tgt.get_viterbi_path(tgt)
            if src_turn:
                bitexts.append(AlignedSent(tgt_viterbi, src_viterbi))
            else:
                bitexts.append(AlignedSent(src_viterbi, tgt_viterbi))
        return bitexts


