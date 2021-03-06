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


class AlignTrainerBase:
    """
        prune stepにalignment lossを加える
    """

    def __init__(self,arg_src, arg_tgt):
        """
            arg_src: UnigramModelのparam
            arg_tgt: UnigramModelのparam
        """

        self.U_src = UnigramModel(arg_src)
        self.U_tgt = UnigramModel(arg_tgt)

        self.n_threads=self.U_src.n_threads

        self.src_tokenised=[]
        self.tgt_tokenised=[]
        print("init")

    def prepare_UnigramModel(self):
        # load sentence
        print("load_sentence")
        self.src_sentences=self.U_src.load_sentence()
        self.tgt_sentences=self.U_tgt.load_sentence()

        # seed_piece
        print("make seed")
        seed_src = self.U_src.make_seed()
        seed_tgt = self.U_tgt.make_seed()

        self.U_src.set_sentence_piece(seed_src)
        self.U_tgt.set_sentence_piece(seed_tgt)


    def align_src_func(self, always_keep, alternatives, freq):
        return self.alignment_loss(always_keep, alternatives, freq,src_turn=True)

    def align_tgt_func(self, always_keep, alternatives, freq):
        return self.alignment_loss(always_keep, alternatives, freq,src_turn=False)


    def prune_step_with_align(self):
        """
        train
        :param: pruneLoss=(1-alpha)*LMloss+(alpha)*AlignLoss
        """

        always_keep_s, alternatives_s = self.U_src.prune_step_1_always_keep_alternative()
        always_keep_t, alternatives_t = self.U_tgt.prune_step_1_always_keep_alternative()

        vsum_s, freq_s, inverted_s = self.U_src.prune_step_2_freq_inverted_pool()
        vsum_t, freq_t, inverted_t = self.U_tgt.prune_step_2_freq_inverted_pool()

        LM_loss_s, new_sentencepieces_s = self.U_src.prune_step_3_new_piece_cand(
            always_keep_s, alternatives_s, vsum_s, freq_s, inverted_s)
        LM_loss_t, new_sentencepieces_t = self.U_tgt.prune_step_3_new_piece_cand(
            always_keep_t, alternatives_t, vsum_t, freq_t, inverted_t)

        start = time.time()
        self.tokenize_viterbi_pool()
        print("time To tokenize=>",time.time()-start)

        align_loss_s = self.align_src_func(always_keep_s,alternatives_s,freq_s)
        align_loss_t = self.align_tgt_func(always_keep_t,alternatives_t,freq_t)

        joint_loss_s = dict()
        joint_loss_t = dict()
        for key in LM_loss_s.keys():
            joint_loss_s[key] = (1-self.alpha)*LM_loss_s[key]+self.alpha*align_loss_s[key]
        for key in LM_loss_t.keys():
            joint_loss_t[key] = (1-self.alpha)*LM_loss_t[key]+self.alpha*align_loss_t[key]

        new_piece_s =self. U_src.prune_step_4_prune_candidate(
            joint_loss_s, new_sentencepieces_s)
        new_piece_t = self.U_tgt.prune_step_4_prune_candidate(
            joint_loss_t, new_sentencepieces_t)

        assert not(self.U_src.SentencePiece.get_piece_size()==len(new_piece_s) and self.U_tgt.SentencePiece.get_piece_size()==len(new_piece_t)),"no piece is  pruned"

        return new_piece_s, new_piece_t

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
            new_piece_src, new_piece_tgt= self.prune_step_with_align()

            self.U_src.set_sentence_piece(new_piece_src)
            self.U_tgt.set_sentence_piece(new_piece_tgt)

            if self.back_up_interval>=1 and step_cnt%self.back_up_interval==0:
                print("back up")
                with open(back_up_file+"_{}.src.pickle".format(step_cnt),"wb") as f:
                    pickle.dump(self.U_src.SentencePiece.get_pieces(),f)
                with open(back_up_file+"_{}.tgt.pickle".format(step_cnt),"wb") as f:
                    pickle.dump(self.U_tgt.SentencePiece.get_pieces(),f)


        print("{} step is needed to converge".format(step_cnt))
        self.U_src.finalize_sentencepiece()
        self.U_tgt.finalize_sentencepiece()

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
        if src_turn:
            for src,tgt in zip(self.src_tokenised,self.tgt_tokenised):
                bitexts.append(AlignedSent(tgt,src))
        else:
            for src,tgt in zip(self.src_tokenised,self.tgt_tokenised):
                bitexts.append(AlignedSent(src,tgt))

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


        if src_turn:
            print("src")
            items=self.U_src.SentencePiece.get_pieces().items()
        else:
            print("tgt")
            items=self.U_tgt.SentencePiece.get_pieces().items()

        losses=[]
        freq_ave=[]
        zero_freq_alt=dict()

        recall_s_key=[]
        for s_key, _ in items:
            #best_alignmentに出てくる数
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
                #print(AlignedWords[s_key])
                #print("key:{} alter{}".format(s_key,alternatives_s[s_key]))
                #print("tgt->src: {}".format(AlignedWords[s_key].items()))

                for t_key, val in AlignedWords[s_key].items():
                    assert val!=0
                    p_t_s = ibm1.translation_table[t_key][s_key]
                    #altのなかで、一番 alignの付け替えられそうなものに付け替える
                    p_t_s_alt = max(
                        ibm1.translation_table[t_key][s_key_alt] for s_key_alt in alternatives_s[s_key])

                    logP_key = log(p_t_s)  # logP(t|x)
                    # P(t|x)がx_altにequally distributed
                    logP_alt = log(p_t_s_alt+p_t_s/len(alternatives_s[s_key]))

                    assert logP_key-logP_alt!=0
                    loss += val/sum_val*(logP_key - logP_alt)
                    losses.append(loss)
                candidate_s[s_key] = loss
                freq_ave.append((freq_s[s_key],sum(freq_s[v] for v in alternatives_s[s_key])/len(alternatives_s[s_key])))
                for v in alternatives_s[s_key]:
                    if freq_s[v]==0:
                        if v in zero_freq_alt:
                            pass
                        else:
                            zero_freq_alt[v]=1
                    else:
                        zero_freq_alt[v]=0


                recall_each=[]
                for w in AlignedWords[s_key].keys():
                    for x in alternatives_s[s_key]:
                        if x in ibm1.translation_table[w].keys():
                            recall_each.append(1)
                            break
                    else:
                        recall_each.append(0)
                if len(recall_each)>0:
                    recall_s_key.append(sum(recall_each)/len(recall_each))


        #print("recall=>",sum(recall_s_key)/len(recall_s_key))
        #print("zero_freq_alt=>",sum(zero_freq_alt.values())/len(zero_freq_alt.keys()))
        #print("ave Frep:{} alter:{}".format(sum(a for a,_ in freq_ave)/len(freq_ave),sum(b for _,b in freq_ave)/len(freq_ave)))


        #print("alignet_cnt= no_alignt:{}/all:{}=use_rate:{}".format(no_align_cnt,all_align_cnt,(all_align_cnt-no_align_cnt)/all_align_cnt))
        #print("loss_avg=>",sum(losses)/len(losses))
        #print("loss_pos=>{}/all{}".format(len(list(filter(lambda x:x>0,losses))),len(losses)))
        #print("end calc of align_loss")
        #input()
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

        use_tgt =[t for i,t in enumerate(self.tgt_sentences) if i in use_idx]
        iterable=[(items,self.U_tgt.SentencePiece.get_pieces(),self.U_tgt.Trie) for items in  zip_longest(*[iter(use_tgt)]*size)]
        print("tgt_viterbi")
        with Pool(processes=self.n_threads) as p:
            tgt_viterbis=p.map(func=process_each,iterable=iterable)



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
