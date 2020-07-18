import sys

def main(f1):
    """
    tokenize fileの文の長さの平均を求める
    """
    avg=0
    cnt=0

    with open(f1) as f:
        for s in f:
            avg+=len(s.split())
            cnt+=1
    print("avg len=>{}".format(avg/cnt))

if __name__=="__main__":
    argv=sys.argv
    assert len(argv)==2,"voc file path"
    main(argv[1])
