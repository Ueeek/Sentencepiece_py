# lattice とUnigramをimport するために必要
from math import log,exp
from Lattice import Lattice
from UnigramModel import UnigramModel
from collections import defaultdict
import pickle


class AlignTrainerBase:
    def __init__(self,arg_src, arg_tgt):

        self.U_src = UnigramModel(arg_src)
        self.U_tgt = UnigramModel(arg_tgt)

    def prepare_UnigramModel(self):
        # load sentence
        self.U_src.load_sentence()
        self.U_tgt.load_sentence()
        # seed_piece
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
            #TODO UnigamModleのEM()とかを呼べばできるのでは
            print("EM")
            self.U_src.run_EM()
            self.U_tgt.run_EM()
            if self.U_src.check_finish() and self.U_tgt.check_finish():
                break
            #for itr in range(2):
            #    # E
            #    exp_src, obj_src, n_token_src = self.U_src.run_e_step()
            #    exp_tgt, obj_tgt, n_token_tgt = self.U_tgt.run_e_step()

            #    # M
            #    new_pieces_src = self.U_src.run_m_step(exp_src)
            #    new_piece_tgt = self.U_tgt.run_m_step(exp_tgt)

            #    self.U_src.set_sentence_piece(new_pieces_src)
            #    self.U_tgt.set_sentence_piece(new_piece_tgt)

            #if self.U_src.SentencePiece.get_piece_size() <= self.U_src.desired_voc_size and self.U_tgt.SentencePiece.get_piece_size() <= self.U_tgt.desired_voc_size:
            #    print("size:SRC:{}.TGT:{}".format(self.U_src.SentencePiece.get_piece_size(),self.U_tgt.SentencePiece.get_piece_size()))
            #    print("desired:SRC:{}.TGT:{}".format(self.U_src.desired_voc_size,self.U_tgt.desired_voc_size))
            #    break

            print("Align")
            new_piece_src, new_piece_tgt= self.prune_step_with_align(alpha=alpha)

            self.U_src.set_sentence_piece(new_piece_src)
            self.U_tgt.set_sentence_piece(new_piece_tgt)

            if back_up_interval>=1 and step_cnt%back_up_interval==0:
                with open(back_up_file+"_{}.src.pickle".format(step_cnt),"wb") as f:
                    pickle.dump(U_src,f)
                with open(back_up_file+"_{}.tgt.pickle".format(step_cnt),"wb") as f:
                    pickle.dump(U_tgt,f)


        print("{} step is needed to converge".format(step_cnt))
        self.U_src.finalize_sentencepiece()
        self.U_tgt.finalize_sentencepiece()

