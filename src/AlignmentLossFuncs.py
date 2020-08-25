from nltk.translate import IBMModel1
from nltk.translate import IBMModel
from nltk.translate import Alignment
from nltk.translate import AlignedSent
from math import log,exp
from Lattice import Lattice
from collections import defaultdict
import pickle

from AlignTrainerBase import AlignTrainerBase

def get_viterbi_path(s, U):
    """
    Arguments:
        s(str) : sentence
        U(class): Unigram Model
    Returns:
        viterbi_tokens(list): list of tokens consists of viterbi path
    """
    L = Lattice()
    L.set_sentence(s)
    L.populate_nodes(U.SentencePiece.get_pieces(), U.Trie)
    viterbi_tokens = L.Viterbi(ret_piece=True)
    return viterbi_tokens

def get_bitexts(U_s,U_t):
    """
    srcとtgtをbest tokenizeしてreturn　する
    Arguments:
        U_s,U_t: Unigram model for source and tgt respectively
    Return
        bitexts(list): text pair for train ibm
    """
    bitexts = []
    for src, tgt in zip(U_s.sentences, U_t.sentences):
        src_viterbi = get_viterbi_path(src, U_s)
        tgt_viterbi = get_viterbi_path(tgt, U_t)
        bitexts.append(AlignedSent(tgt_viterbi, src_viterbi))
    return bitexts

def alignment_loss(U_s, U_t, always_keep_s, alternatives_s, freq_s):
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

    bitexts = get_bitexts(U_s,U_t)
    # Train IBM Model1 with best tokenize sentence of source and target(bitext,iteration)
    ibm1 = IBMModel1(bitexts, 2)


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
    for s_key, _ in U_s.SentencePiece.get_pieces().items():
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
    return candidate_s
