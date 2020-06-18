# lattice とUnigramをimport するために必要
#TODO 普段使うなって感じだけど、align_lossの計算をいじるために、新しくコピーして作ったファイル。あとで消せ
from nltk.translate import IBMModel1
from nltk.translate import IBMModel
from nltk.translate import Alignment
from nltk.translate import AlignedSent
from math import log
from Lattice import Lattice
from UnigramModel import UnigramModel
from collections import defaultdict


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

def no_alignment_loss(U_s, U_t, always_keep_s, alternatives_s, freq_s):
    return defaultdict(int)

def alignment_loss_all_alignment(U_s, U_t, always_keep_s, alternatives_s, freq_s):
    """ alignlossを求めたい
    U_sにalignment lossを加える
    X,Yはbest, Aは全てのAについて試す。(bestAだとsparseになる)

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

    # AlignedWords[key1][key2]=val, key1をsrcの語とする、key2はkey1とval回共起するtgtの語
    CoocWords = defaultdict(lambda: defaultdict(int))
    for bitext in bitexts:
        # align=(idx_in_tgt,idx_in_src)
        tgt, src = bitext.words, bitext.mots
        for s in src:
            for t in tgt:
                CoocWords[s][t] += 1

    # Coocするならalignの可能性があり、cooc_freq*P(t|x)でlossを計算する。
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
            if len(CoocWords[s_key].items()) == 0:
                print(s_key)
                print(CoocWords[s_key].items())
                exit()
                no_align_cnt += 1

            sum_val = sum(CoocWords[s_key].values())
            for t_key, val in CoocWords[s_key].items():
                p_t_s = ibm1.translation_table[t_key][s_key]
                p_t_s_alt = max(
                    ibm1.translation_table[t_key][s_key_alt] for s_key_alt in alternatives_s[s_key])

                #p_alt = p_t_s_alt+p_t_s/len(alternatives_s[s_key])
                p_alt = p_t_s_alt
                logP_key = log(p_t_s)  # logP(t|x)
                # P(t|x)がx_altにequally distributed
                logP_alt = log(p_alt)
                loss += val/sum_val*(logP_key - logP_alt)
            candidate_s[s_key] = loss
    #print("zero_rate =>", sum(v == 0 for v in candidate_s.values())/len(candidate_s))
    #print("no_align_rate=>", no_align_cnt/all_align_cnt)
    #print("top_10=>",list(sorted(AlignedCnt.items(),key=lambda x:-x[1]))[:10])
    return candidate_s


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

    #print("sum_t P(t,s)=>",[sum(ibm1.translation_table[tt].values())-ibm1.translation_table[tt][None] for tt in ibm1.translation_table.keys()])
    #print("sum_s P(t,s)=>",[sum(ibm1.translation_table[tt].values())-ibm1.translation_table[tt][None] for tt in ibm1.translation_table.keys()])

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
    #print("zero_rate=>",sum(v==0 for v in candidate_s.values())/len(candidate_s))
    # print("no_align_rate=>",no_align_cnt/all_align_cnt)
    #print("top_10=>",list(sorted(AlignedCnt.items(),key=lambda x:-x[1]))[:10])
    # for key,val in list(sorted(AlignedCnt.items(),key=lambda x:-x[1]))[:20]:
    #    print("word:{}->{}".format(key,AlignedWords[key].items()))
    return candidate_s

def prune_step_with_align(U_s,U_t,src_func,tgt_func=None):
    """
        引数でどうやってalignlossを計算するかを切り替えたい
    Arguments:
        src_func: U_sのalign lossを計算する関数
        tgt_func:　src_funcを同様。Noneなら、src_funcと同じものとする
    """
    if tgt_func is None:
        tgt_func = src_func

    always_keep_s, alternatives_s = U_s.prune_step_1_always_keep_alternative()
    always_keep_t, alternatives_t = U_t.prune_step_1_always_keep_alternative()

    vsum_s, freq_s, inverted_s = U_s.prune_step_2_freq_inverted()
    vsum_t, freq_t, inverted_t = U_t.prune_step_2_freq_inverted()

    LM_loss_s, new_sentencepieces_s = U_s.prune_step_3_new_piece_cand(
        always_keep_s, alternatives_s, vsum_s, freq_s, inverted_s)
    LM_loss_t, new_sentencepieces_t = U_t.prune_step_3_new_piece_cand(
        always_keep_t, alternatives_t, vsum_t, freq_t, inverted_t)

    align_loss_s = src_func(U_s,U_t,always_keep_s,alternatives_s,freq_s)
    align_loss_t = tgt_func(U_t,U_s,always_keep_t,alternatives_t,freq_t)

    joint_loss_s = dict()
    joint_loss_t = dict()
    for key in LM_loss_s.keys():
        joint_loss_s[key] = LM_loss_s[key]+align_loss_s[key]
    for key in LM_loss_t.keys():
        joint_loss_t[key] = LM_loss_t[key]+align_loss_t[key]

    new_piece_s = U_s.prune_4_prune_candidate(
        joint_loss_s, new_sentencepieces_s)
    new_piece_t = U_t.prune_4_prune_candidate(
        joint_loss_t, new_sentencepieces_t)
    assert not(U_s.SentencePiece.get_piece_size()==len(new_piece_s) and U_t.SentencePiece.get_piece_size()==len(new_piece_t)),"no piece is  pruned"
    return new_piece_s, new_piece_t

def train_align(arg_src, arg_tgt, alter=False,allA=False):
    """
    Arguments:
        alter(bool): false ならsrcとtgt、同じステップで両方ともpruneでalinを考慮する・
        trueなら、srcとtgtでalignmの考慮を交互にする(隔step)
    """
    print("Train align")
    U_src = UnigramModel(arg_src)
    U_tgt = UnigramModel(arg_tgt)

    # load sentence
    U_src.load_sentence()
    U_tgt.load_sentence()
    # seed_piece
    seed_src = U_src.make_seed()
    #seed_src = U_src.make_seed_sentence_piece()
    U_src.set_sentnece_piece(seed_src)

    seed_tgt = U_tgt.make_seed()
    #seed_tgt = U_tgt.make_seed_sentence_piece()
    U_tgt.set_sentnece_piece(seed_tgt)

    # Start EM
    print("Seed voc size=> src:{} tgt:{}\nStart EM training".format(U_src.SentencePiece.get_piece_size(),U_tgt.SentencePiece.get_piece_size()))
    step_cnt = 0
    while True:
        step_cnt += 1
        for itr in range(2):
            # E
            exp_src, obj_src, n_token_src = U_src.run_e_step()
            exp_tgt, obj_tgt, n_token_tgt = U_tgt.run_e_step()

            # M
            new_pieces_src = U_src.run_m_step(exp_src)
            new_piece_tgt = U_tgt.run_m_step(exp_tgt)

            # update
            U_src.set_sentnece_piece(new_pieces_src)
            U_tgt.set_sentnece_piece(new_piece_tgt)

            print("EN EM sub_iter= {} size={} obj={} num_tokens= {} num_tokens/piece= {}".format(itr,
                                                                                                 U_src.SentencePiece.get_piece_size(), obj_src, n_token_src, n_token_src/U_src.SentencePiece.get_piece_size()))
            print("JA EM sub_iter= {} size={} obj={} num_tokens= {} num_tokens/piece= {}".format(itr,
                                                                                                 U_tgt.SentencePiece.get_piece_size(), obj_tgt, n_token_tgt, n_token_tgt/U_tgt.SentencePiece.get_piece_size()))
        if U_src.SentencePiece.get_piece_size() <= U_src.desired_voc_size and U_tgt.SentencePiece.get_piece_size() <= U_tgt.desired_voc_size:
            break

        if alter:
            if step_cnt % 2:#srcのみ
                new_piece_src, new_piece_tgt = prune_step_with_align(U_src,U_tgt,alignment_loss,no_alignment_loss)
            else:#tgtのみ
                new_piece_src, new_piece_tgt = prune_step_with_align(U_src,U_tgt,no_alignment_loss,alignment_loss)
        else:
            if allA:
                new_piece_src, new_piece_tgt = prune_step_with_align(U_src,U_tgt,alignment_loss_all_alignment)
            else:
                new_piece_src, new_piece_tgt = prune_step_with_align(U_src,U_tgt,alignment_loss)


        U_src.set_sentnece_piece(new_piece_src)
        U_tgt.set_sentnece_piece(new_piece_tgt)

    print("{} step is needed to converge".format(step_cnt))
    U_src.finalize_sentencepiece()
    U_tgt.finalize_sentencepiece()