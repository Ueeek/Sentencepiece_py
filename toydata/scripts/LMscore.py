import sys
sys.path.append("../../src")
from UnigramModel import UnigramModel

def main(tokenized_sent,voc_file_path):
    D=dict()
    with open(voc_file_path) as f:
        for s in f:
            v = s.split("\t")
            D[v[0]]=float(v[1])

    score=0
    all_cnt=0
    with open(tokenized_sent) as f:
        for s in tokenized_sent:
            for c in s.split():
                all_cnt+=1
                score+=D[c]
    print(score)
    print(score/all_cnt)


if __name__=="__main__":
    argv=sys.argv
    assert len(argv)==3,"requiereed 2 params :: corpus_path,voc_file_path"
    main(*argv[1:])



