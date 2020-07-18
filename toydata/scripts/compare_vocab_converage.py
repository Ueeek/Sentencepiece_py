import sys

def main(file1,file2,cut_ratio="1",tail=False):
    """
    f1とf2のvocabの共通数を見る
    cut_ratioは上位cur_ratio%だけを比較する
    """

    cut_ratio=float(cut_ratio)

    vocs1=[]
    vocs2=[]
    with open(file1) as f1,open(file2) as f2:
        for s in f1:
            s = s.replace(chr(9601),"_")
            vocs1.append(s.split("\t")[0])
        for s in f2:
            s = s.replace(chr(9601),"_")
            vocs2.append(s.split("\t")[0])
    if tail:
        voc1={v for v in vocs1[int(len(vocs1)*cut_ratio):]}
        voc2={v for v in vocs2[int(len(vocs2)*cut_ratio):]}
    else:
        voc1={v for v in vocs1[:int(len(vocs1)*cut_ratio)]}
        voc2={v for v in vocs2[:int(len(vocs2)*cut_ratio)]}

    print("voc1 size=>{}".format(len(voc1)))
    print("voc2 size=>{}".format(len(voc2)))

    print("voc1 and voc2 size=>{}".format(len(voc1 & voc2)))
    print("voc1 only =>{}".format(len(voc1-voc2)))
    print("voc2 only =>{}".format(len(voc2-voc1)))

    print("voc1 converage =>{}".format(len(voc1 & voc2)/len(voc1)))
    print("voc2 converage =>{}".format(len(voc1 & voc2)/len(voc2)))



if __name__=="__main__":
    argv = sys.argv
    assert 3<=len(argv)<=5
    main(*argv[1:])
