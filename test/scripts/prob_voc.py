import sys
from math import exp


def main(file1,file2,cut_ratio="1",tail=False):
    """
    f1とf2の同じvocのlog_probの違いを見る
    cut_ratioは上位cur_ratio%だけを比較する
    """

    cut_ratio=float(cut_ratio)

    vocs1=[]
    vocs2=[]
    d_1=dict()
    d_2=dict()
    with open(file1) as f1,open(file2) as f2:
        for s in f1:
            key,val = s.split("\t")
            d_1[key]=float(val)
            
        for s in f2:
            key,val = s.split("\t")
            d_2[key]=float(val)

    common_voc=d_1.keys() & d_2.keys()

    log_prob_diff_sum= sum(abs(d_1[key]-d_2[key]) for key in common_voc)
    print(log_prob_diff_sum/len(common_voc))
    print(exp(-log_prob_diff_sum)/len(common_voc))




if __name__=="__main__":
    argv = sys.argv
    assert 3<=len(argv)<=5
    main(*argv[1:])
