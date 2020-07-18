from collections import defaultdict
import sys
def main(f1,f2):
    """
    二つのtokenized fileの文対で、unigramの共通率を調べる

    共通率は、s1のtokenの中で、s1にも現れるのは何%か　を各文でやって平均
    """

    sentneces1=[]
    sentneces2=[]
    with open(f1) as f:
        for s in f:
            sentneces1.append(s.split())
    with open(f2) as f:
        for s in f:
            sentneces2.append(s.split())

    assert len(sentneces2)==len(sentneces1)
    for s,t in zip(sentneces1,sentneces2):
        assert "".join(s)=="".join(t),"\n{}\n{}".format(s,t)

    rate1=0
    rate2=0

    for s1,s2 in zip(sentneces1,sentneces2):
        D1=defaultdict(int)
        D2=defaultdict(int)
        for w1 in s1:
            D1[w1]+=1
        for w2 in s2:
            D2[w2]+=1
        rate1 += sum([min(D1[key],D2[key]) for key in D1.keys()])/len(s1)
        rate2 += sum([min(D2[key],D1[key]) for key in D2.keys()])/len(s2)

    rate1/=len(sentneces1)
    rate2/=len(sentneces2)
    
    print("rate1=>{}\nrate2=>{}".format(rate1,rate2))

if __name__=="__main__":
    argv = sys.argv
    main(*argv[1:])
