from util import LOG,UTF8ToUnicodeText,UnicodeCharToUTF8
from collections import defaultdict
from pysuffixarray.core import SuffixArray
from SentencePiece import SentencePiece
from math import log

class UnigramModel:
    """どこまで仕事をするのか
    """
    def __init__(self, argv):
        """ get parameter from argv
        """
        self.SrcSentencePiece=SentencePiece()
        self.TgtSentencePiece=SentencePiece()

        self.src_file=argv["src_file"]
        self.tgt_file=argv["tgt_file"]


        self.src_sentences=[]
        self.tgt_sentences=[]
        
        self.seed_sentence_piece_size=100


    def make_seed_sentence_piece(self,sentences:list):
        """ set init vocabulary of sentence piece
        """

        all_chars=defaultdict(int)
        array=[]

        #Merge all sentences into one array with 0x0000 delimiter
        #TODO 日本語はSAに対応してないからunicodeでやる必要があるが、とりま英語でやる
        kSentenceBoundary = chr(0x0000);

        for s in sentences:
            for word in s.split():
                word="_"+word
                for c in word:
                    array.append(c)
                    if c!=kSentenceBoundary:
                        all_chars[c]+=1
                array.append(kSentenceBoundary)
                

        #make a suffix_array to extract all sub strings occuring more than 2 times in the sentence
        print("Making Suffix Array")
        A = "".join(array)
        SA = SuffixArray(A)

        print("Extracting frequent sub strings...")
        # TODO 結構怪しい気がする ここの処理
        substr=[]
        for i,l in enumerate(SA.longest_common_prefix()):
            if l<=1:#lcp=1なので1回しか出てこない
                continue
            sb =SA.string[SA.sa[i]:SA.sa[i]+l]#2回以上出てくるsbst
            if sb[-1]==kSentenceBoundary:#最後の "0x00"は大目に見る
                sb=sb[:-1]
            if len(sb)<=1:#多目に見た後に長さが2.elseはsb=char担ってる時
                continue
            if any(v==kSentenceBoundary for v in sb):#途中に 0x00が入っているのはinvalid
                continue

            #それでも残ったやつは、2回以上出てくるsbst
            freq = SA.match(sb)
            substr.append((sb,len(sb)*len(freq)))

        subtr = sorted(substr,key=lambda x:-x[1])
        seed_sentencepieces=all_chars
        if len(seed_sentencepieces)>self.seed_sentence_piece_size:
            pass
        elif len(seed_sentencepieces)+len(substr)>self.seed_sentence_piece_size:
            delete = self.seed_sentence_piece_size - len(seed_sentencepieces)-len(substr)
            print("del {} freq-sbst because of seed_sentence_piece_size".format(delete))
            for sb,val in substr[:delete]:
                seed_sentencepieces[sb]=val
        else:
            for sb,val in substr:
                seed_sentencepieces[sb]=val

        
        #freqを確率として扱うためにsum=1にする。その後log probにするをまとめてやってる
        s=log(sum([v for v in seed_sentencepieces.values()]))
        for i,v in seed_sentencepieces.items():
            seed_sentencepieces[i]=log(v)-s

        for i,v in seed_sentencepieces.items():
            print("i,v=>",i,v)
        return seed_sentencepieces




    def __set_sentnece_piece(self,piece):
        """ set piece
        """
        pass
    def __load_sentence(self):
        """ load sentence from file
        """
        src=[]
        tgt=[]
        with open(self.src_file) as src_f:
            for s in src_f:
                src.append(s)
        with open(self.tgt_file) as tgt_f:
            for t in tgt_f:
                tgt.append(t)

        assert len(src)!=len(tgt)
        self.src_sentences=src
        self.tgt_sentences=tgt


        pass


    def train(self):
        self.__load_sentence()
        seed_sentencepieces = make_seed_sentence_piece()
        self.__set_sentnece_piece("a")

        while True:
            for _ in range(10):#EM iteration loop
                self.__run_e_step()
                self.__run_m_step()
            self.__prune_vocabulary()




if __name__=="__main__":
    dummy_arg={"src_file":None,"tgt_file":None}
    U = UnigramModel(dummy_arg)
    #sentences=["I am happy","You may be happy","私は元気","歩くの大好き@./,;1"]
    sentences=["I am happy","You may be happy","I my me may be"]
    U.make_seed_sentence_piece(sentences)

