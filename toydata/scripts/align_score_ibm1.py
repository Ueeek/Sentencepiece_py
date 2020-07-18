from nltk.translate import IBMModel1
from nltk.translate import IBMModel
from nltk.translate import Alignment
from nltk.translate import AlignedSent 
from collections import defaultdict
from math import log,exp
import sys

def main(src_path,tgt_path):
    srcs=[]
    tgts=[]
    with open(src_path) as f:
        for s in f:
            srcs.append(s.split())
    with open(tgt_path) as f:
        for t in f:
            tgts.append(t.split())

    assert len(srcs)==len(tgts)

    bitexts_s2t = []
    bitexts_t2s = []
    for s,t in zip(srcs,tgts):
        bitexts_s2t.append(AlignedSent(t, s))
        bitexts_t2s.append(AlignedSent(s,t))

    ibm1_s2t = IBMModel1(bitexts_s2t,5)
    ibm1_t2s = IBMModel1(bitexts_t2s,5)

    p_s_given_t =0
    p_t_given_s =0
    Ds=defaultdict(list)
    Dt=defaultdict(list)
    Dlen=defaultdict(list)
    Dscore=defaultdict(list)

    for key in ibm1_t2s.translation_table.keys():
        #print("sum t[t|*]=>",sum(ibm1_t2s.translation_table[key].values()))
        for key_s in ibm1_t2s.translation_table[key].keys():
            Ds[key_s].append(ibm1_t2s.translation_table[key][key_s])

    for key in Ds.keys():
        #print("sum P(*|s)=>",sum(Ds[key]))
        pass

    for key in Ds.keys():
        if key is None:
            continue
        Dlen[len(key)].append(len(Ds[key]))
        Dscore[len(key)].extend(Ds[key])

    for key in sorted(Dlen.keys(),key=lambda x:int(x)):
        #print("Dlen=>",key,sum(Dlen[key])/len(Dlen[key]))
        #print("Dscore=>",key,sum(Dscore[key])/len(Dscore[key]))
        pass

    for b in bitexts_t2s:
        tgt,src,align = b.words,b.mots,b.alignment
        for (idx_tgt, idx_src) in align:
            if idx_src is None:
                continue
            #print("t:{}->s:{}".format(tgt[idx_tgt],src[idx_src]))
            p_t_given_s +=  log(ibm1_t2s.translation_table[tgt[idx_tgt]][src[idx_src]])

    for b in bitexts_s2t:
        src,tgt,align = b.words,b.mots,b.alignment
        for (idx_src, idx_tgt) in align:
            if idx_tgt is None:
                continue
            p_s_given_t +=  log(ibm1_s2t.translation_table[src[idx_src]][tgt[idx_tgt]])

    p_s_given_t=p_s_given_t/sum([len(v) for v in tgts])
    p_t_given_s=p_t_given_s/sum([len(v) for v in srcs])

    print("log P(s|t)=> {}\nlog P(t|s)=>{}".format(p_s_given_t,p_t_given_s))



if __name__=="__main__":
    argv=sys.argv
    main(*argv[1:])

