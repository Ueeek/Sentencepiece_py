import sys

def main(file1,file2,cut_ratio="1",tail=False):
    """
    f1とf2のvocabの異なるvocを見る
    cut_ratioは上位cur_ratio%だけを比較する
    """

    cut_ratio=float(cut_ratio)

    vocs1=[]
    vocs2=[]
    with open(file1) as f1,open(file2) as f2:
        for s in f1:
            vocs1.append(s.split("\t")[0])
        for s in f2:
            vocs2.append(s.split("\t")[0])
    if tail:
        voc1={v for v in vocs1[int(len(vocs1)*cut_ratio):]}
        voc2={v for v in vocs2[int(len(vocs2)*cut_ratio):]}
    else:
        voc1={v for v in vocs1[:int(len(vocs1)*cut_ratio)]}
        voc2={v for v in vocs2[:int(len(vocs2)*cut_ratio)]}

    print("voc1_only=>",voc1-voc2)
    print("voc2_only=>",voc2-voc1)

    print("len_avg1=>",sum(len(v) for v in (voc1-voc2))/len(voc1-voc2))
    print("len_avg2=>",sum(len(v) for v in (voc2-voc1))/len(voc2-voc1))



if __name__=="__main__":
    argv = sys.argv
    assert 3<=len(argv)<=5
    main(*argv[1:])
