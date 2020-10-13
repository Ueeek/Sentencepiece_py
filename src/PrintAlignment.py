from translate import IBMModel1, IBMModel, Alignment, AlignedSent
from collections import defaultdict
from math import log,exp
import sys

def main(src_path,tgt_path):
    """
    print alignment
    from t -> s
    """
    #list ofsentence.split(): 
    srcs=[]
    tgts=[]

    with open(src_path) as f:
        for s in f:
            srcs.append(s.split())
    with open(tgt_path) as f:
        for t in f:
            tgts.append(t.split())

    assert len(srcs)==len(tgts)

    bitexts= []
    for s,t in zip(srcs,tgts):
        bitexts.append(AlignedSent(s,t))
        #bitexts.append(AlignedSent(t,s))

    ibm1 = IBMModel1(bitexts,5)

    for b in bitexts:
        ret=[]
        align=[]
        for idx_src,idx_tgt in b.alignment:
            assert idx_src is not None
            align.append((idx_src,idx_tgt))

        for idx_src, idx_tgt in sorted(align, key=lambda x:x[0]):
            ret.append("{}-{}".format(idx_tgt,idx_src))
        print(" ".join(ret))


if __name__=="__main__":
    argv=sys.argv
    main(*argv[1:])

