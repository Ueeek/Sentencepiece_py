import sys
import matplotlib.pyplot as plt
from collections import defaultdict

def main(voc_file):
    """
        voc fileのvocのlengthのdistributonをgraphにする
    """
    print("file:",voc_file)
    MAX_LEN=20
    D=defaultdict(int)
    with open("../voc/"+voc_file) as f:
        for s in f:
            voc,val = s.split("\t")
            D[len(voc)]+=1

    X=list(range(MAX_LEN+1))
    Y = [D[x] for x in X]

    #plt.hist(Y,bins=1)
    plt.bar(X,Y)
    plt.title(label=voc_file)
    plt.savefig("../voc_distri_fig/"+voc_file+".distri.png")
    plt.show()


if __name__=="__main__":
    argv = sys.argv
    main(argv[1])

