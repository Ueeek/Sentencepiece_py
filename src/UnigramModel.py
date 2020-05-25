from collections import defaultdict
from pysuffixarray.core import SuffixArray
from SentencePiece import SentencePiece
from math import log
from Lattice import Lattice
import pygtrie as trie
from util import *

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

        self.src_words=[]
        self.tgt_words=[]
        
        self.seed_sentence_piece_size=10000


    def __make_seed_sentence_piece(self,key="src"):
        """ set init vocabulary of sentence piece
        """

        all_chars=defaultdict(int)
        array=[]

        #Merge all sentences into one array with 0x0000 delimiter
        #TODO 日本語はSAに対応してないからunicodeでやる必要があるが、とりま英語でやる
        kSentenceBoundary = chr(0x0000);

        if key=="src":
            sentences=self.src_sentences
        elif key=="tgt":
            sentences=self.tgt_sentences

        for s in sentences:
            for word in s.split("_"):
                if len(word)==0:#文の先頭に"_"がついているからsplit[0]は""になる
                    continue
                word="_"+word
                for c in word:
                    uni_c = UTF8ToUnicodeText(c)
                    c = UnicodeCharToUTF8(uni_c)
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
        #sbstrでduplicateがあったからsetにしとく(sb[-1]=boundのとこで、a+bound,aの二つがdup?)
        substr=set()
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
            substr.add((sb,len(sb)*len(freq)))

        #print("all char:{}, substr:{}".format(len(all_chars),len(substr)))
        #print("all_chars=>"," ".join(list(sorted(all_chars.keys(),))))
        #print("freq_sbstr:=>"," ".join(sorted([v[0] for v in substr])))
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

        return seed_sentencepieces




    def __set_sentnece_piece(self,pieces,key="src"):
        """ set piece
        """

        assert key in {"src","tgt"}
        if key=="src":
            self.SrcSentencePiece.set_sentence_piece(pieces)
        elif key=="tgt":
            self.TgtSentencePiece.set_sentence_piece(pieces)

    def __load_sentence(self,key="src"):
        """ load sentence from file
        replace \s as "_"
        """
        if key=="src":
            path=self.src_file
        elif key=="tgt":
            path=self.tgt_file
        print("loading sentences\n {}:{}".format(key,path))

        sentences=[]
        words=defaultdict(int)
        with open(path) as f:
            for s in f:
                _s = "_"+"_".join(s.split())
                for w in s.split():
                    words["_"+w]+=1

                sentences.append(_s)

        if key=="src":
            self.src_sentences = sentences
            self.src_words=words
        elif key=="tgt":
            self.tgt_sentences = sentences

    def __run_e_step(self,key="src"):
        """E step of EM learnign
        Return:
            objective: int
            nun_token: int
            expected: float[len(piece)]
        """
        print("E step")
        #TODO とりあえず srcのみ
        expected=defaultdict(int)
        objective=0
        num_tokens=0

        all_sentence_freq=sum(self.src_words.values())
        print("all_sentence_freq=>",len(self.src_sentences))
        print("all_words_freq=>",all_sentence_freq)

        #for i in range(len(self.src_sentences)):
        for key,freq in sorted(self.src_words.items()):
            L = Lattice()
            L.set_sentence(key)
            L.populate_nodes(self.SrcSentencePiece.get_pieces())
            Z,ret_expected = L.populate_marginal(freq)

            for key,val in ret_expected.items():
                expected[key]+=val

            N = len(L.Viterbi())
            num_tokens+=N
            objective -= Z/all_sentence_freq

            #s = self.src_sentences[i]
            #freq =1 #sentenceごとに見るから(remove?)
            #for word in s.split("_"):
            #    if len(word)==0: continue
            #    word="_"+word
            #    L = Lattice()
            #    L.set_sentence(word)
            #    L.populate_nodes(self.SrcSentencePiece.get_pieces())

            #    Z,ret_expected = L.populate_marginal()
            #    #ret_expected is dict => aggregate value into expected(list)
            #    for key,val in ret_expected.items():
            #        expected[key]+=val

            #    N = len(L.Viterbi())
            #    num_tokens+=N
            #    objective-=Z/all_sentence_freq
        return expected,objective,num_tokens

    def __run_m_step(self,expected,key="src"):
        """ M step of EM learning
        Return:
            new_sentencepieces: list of sentencepiece
        """
        print("Run M step")
        current_piece = self.SrcSentencePiece.get_pieces()

        assert len(current_piece)==len(expected)

        new_pieces=defaultdict(int)
        sum_freq=0
        #filter infrequent sentencepieces here
        #TODO 1 charが消されることで tokenizeできなくなっている
        #expectedの値とかが小さすぎてる?
        for key,val in current_piece.items():
            freq = expected[key]
            kExpectedFrequencyThreshold=0.5

            if freq<kExpectedFrequencyThreshold:
                #assert len(key)!=1 ,print("invalid removal 1 char=>{} freq=>{}".format(key,freq))
                continue
            new_pieces[key]=freq
            sum_freq+=freq
        print("filtered infrequent sentencepiece, {} pieces removed".format(len(current_piece)-len(new_pieces)))

        logsum=Digamma(sum_freq)
        for key,val in new_pieces.items():
            new_pieces[key] = Digamma(val)-logsum

        return new_pieces


    def __prune_piece(self):
        current_piece = self.SrcSentencePiece.get_pieces()

        always_keep=[True]*len(current_piece)
        alternatives=[[] for _ in range(len(current_piece))]

        for i,(piece,score) in enumerate(current_piece.items()):
            L = Lattice()
            L.set_sentence(piece)
            L.populate_nodes(self.SrcSentencePiece.get_pieces())
           # print("surface_0=>","".join([L.nodes[v].piece for v in nbests[0]]))
            if len(nbests)==2:
                print("surface_1=>","".join([L.nodes[v].piece for v in nbests[1]]))

            if len(nbests)==1:
                always_keep[i]=True
            elif len(nbests[0])>=2:
                always_keep[i]=False
            elif len(nbests[0])==1:
                always_keep[i]=True
                for node in nbests[1]:
                    pass


    def build_trie(self):
        pass
        

    def train(self):
        #src
        self.__load_sentence(key="src")
        seed_sentencepieces = self.__make_seed_sentence_piece(key="src")
        self.__set_sentnece_piece(seed_sentencepieces,key="src")


        self.SrcSentencePiece.print_piece()
        print("seed_")
        for key,val in self.SrcSentencePiece.get_pieces().items():
            print("key=> {} score=> {}".format(key,val))


        #while True:
        for _ in range(5):
            for itr in range(1):#EM iteration loop
                #print("piece=>",self.SrcSentencePiece.get_pieces())
                expected,objective,num_tokens = self.__run_e_step()

                #for key,val in self.SrcSentencePiece.get_pieces().items():
                #    print("key=> {} score=> {} exp=>{}".format(key,val,expected[key]))
                new_sentencepieces = self.__run_m_step(expected)
                self.SrcSentencePiece.set_sentence_piece(new_sentencepieces)
                print("EM sub_iter= {} size={} obj={} num_tokens= {} num_tokens/piece= {}".format(itr,self.SrcSentencePiece.get_piece_size(),objective,num_tokens,num_tokens/self.SrcSentencePiece.get_piece_size()))
                exit()

            self.__prune_piece()




if __name__=="__main__":
    dummy_arg={"src_file":"../test/dummy.en","tgt_file":None}
    U = UnigramModel(dummy_arg)
    U.train()
