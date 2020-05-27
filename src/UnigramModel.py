from collections import defaultdict
from SentencePiece import SentencePiece
from math import log
from Lattice import Lattice
import pygtrie
from util import *
from pysuffixarray.core import SuffixArray


class UnigramModel:
    """どこまで仕事をするのか
    """
    def __init__(self, argv):
        """ get parameter from argv
        """
        self.file=argv["file"]
        self.out_voc_file=argv["voc"]
        self.shrinking_rate =argv["shrinking_rate"]
        self.desired_voc_size = argv["desired_voc_size"]
        self.seed_sentence_piece_size = argv["seed_sentence_piece_size"]


        self.SentencePiece=SentencePiece()
        self.Trie=None
        self.sentences=[]
        self.words=[]


    def make_seed_sentence_piece(self):
        """ set init vocabulary of sentence piece

        Return:
            seed_sentencepieces(dict): dict[piece]=score
        """

        all_chars=defaultdict(int)
        array=[]

        #Merge all sentences into one array with 0x0000 delimiter
        kSentenceBoundary = chr(0x0000);

        for s in self.sentences:
            #ここでpretolenizeってのをかましている
            for word in s.split("_"):
                if len(word)==0:#文の先頭に"_"がついているからsplit[0]は""になる
                    continue
                word="_"+word
                for c in word:
                    #really needed?
                    uni_c = UTF8ToUnicodeText(c)
                    c = UnicodeCharToUTF8(uni_c)
                    array.append(c)
                    if c!=kSentenceBoundary:
                        all_chars[c]+=1
                array.append(kSentenceBoundary)

        print("alphabet=>",len(all_chars))
        #print(" ".join(list(sorted(all_chars.keys()))))
                
        #make a suffix_array to extract all sub strings occuring more than 2 times in the sentence
        print("Making Suffix Array")
        A = "".join(array)
        SA = SuffixArray(A)

        print("Extracting frequent sub strings...")
        # TODO 結構怪しい気がする ここの処理
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

        substr = sorted(list(substr),key=lambda x:-x[1])
        seed_sentencepieces=all_chars
        if len(seed_sentencepieces)>self.seed_sentence_piece_size:
            pass
        elif len(seed_sentencepieces)+len(substr)>self.seed_sentence_piece_size:
            delete_size = self.seed_sentence_piece_size - len(seed_sentencepieces)-len(substr)
            print("del {} freq-sbst because of seed_sentence_piece_size".format(delete_size))
            for sb,val in substr[:delete_size]:
                seed_sentencepieces[sb]=val
        else:
            for sb,val in substr:
                seed_sentencepieces[sb]=val

        
        s=log(sum([v for v in seed_sentencepieces.values()]))
        for i,v in seed_sentencepieces.items():
            seed_sentencepieces[i]=log(v)-s

        return seed_sentencepieces




    def set_sentnece_piece(self,pieces):
        """ set piece into Sentencepiece class
        Always call build_trie to create new Trie corresponding to new_pieces
        Args:
            pieces(dict): current sentencepieces dict[piece]=score
        """

        self.SentencePiece.set_sentence_piece(pieces)
        self.build_trie(pieces)

    def load_sentence(self):
        """ load sentence from file
        """
        path=self.file
        print("loading sentences\n {}".format(path))

        sentences=[]
        words=defaultdict(int)
        with open(path) as f:
            for s in f:
                #_s = "_"+"_".join(s.split(" "))#全角と半角のspaceを区別するか(\tとか\nもsplitされるs.split())
                _s = "_"+"_".join(s.split())
                for w in s.split():
                    words["_"+w]+=1

                sentences.append(_s)

        self.sentences = sentences
        self.words=words

    def run_e_step(self):
        """E step of EM learning
        Return:
            objective(int): int
            nun_token(int): sum of the token num of Viterbi path
            expected(dict): dict[piece]=score of the piece
        """
        print("E step")
        #TODO とりあえず のみ
        expected=defaultdict(int)
        objective=0
        num_tokens=0

        all_sentence_freq=sum(self.words.values())

        for key,freq in sorted(self.words.items()):
            L = Lattice()
            L.set_sentence(key)
            L.populate_nodes(self.SentencePiece.get_pieces(),self.Trie)
            Z,ret_expected = L.populate_marginal(freq)

            for key,val in ret_expected.items():
                expected[key]+=val

            N = len(L.Viterbi())
            num_tokens+=N
            objective -= Z/all_sentence_freq

        return expected,objective,num_tokens

    def run_m_step(self,expected):
        """ M step of EM learning
        Return:
            new_sentencepieces: list of sentencepiece
        """

        print("Run M step")
        current_piece = self.SentencePiece.get_pieces()
        assert len(current_piece)>=len(expected)

        new_pieces=dict()
        sum_freq=0
        #filter infrequent sentencepieces here
        for key,val in current_piece.items():
            freq = expected[key]
            kExpectedFrequencyThreshold=0.5

            if freq<kExpectedFrequencyThreshold:
                continue
            new_pieces[key]=freq
            sum_freq+=freq
        print("filtered infrequent sentencepiece, {} pieces removed".format(len(current_piece)-len(new_pieces)))

        logsum=Digamma(sum_freq)
        for key,val in new_pieces.items():
            new_pieces[key] = Digamma(val)-logsum

        return new_pieces


    def prune_piece(self):
        current_piece = self.SentencePiece.get_pieces()

        #pieceをkeyとしてdictで管理
        always_keep=dict()
        alternatives=defaultdict(list)

        #First segments the current sentencepieces to kwon how each sentencepiece is resegmented if this sentencepiece is  removed from vocabulary.
        for key,score in current_piece.items():
            L = Lattice()
            L.set_sentence(key)
            L.populate_nodes(current_piece,self.Trie)
            nbests = L.NBest(2)

            if len(nbests)==1:#only one way to resegment it
                always_keep[key]=True

            elif len(nbests[0])>=2:
                always_keep[key]=False

            elif len(nbests[0])==1:
                always_keep[key]=True
                alternatives[key]=nbests[1]

        #Second, segments all sentences to compute likelihoood with a Unigram LM
        #inverted[key] stires the set of sentence index where the sentencepiece (key) appears
        vsum=0
        freq=defaultdict(int) 
        inverted=defaultdict(int)

        for s,score in self.words.items():
            vsum+=score
            L.set_sentence(s)
            L.populate_nodes(current_piece,self.Trie)

            for node_id in L.Viterbi():
                word = L.nodes[node_id].piece
                if node_id>0:
                    #TODO what is difference of freq and inverted
                    freq[word] += score
                    inverted[word]+=score

        #calc loss
        sum_freq = sum(freq.values())
        logsum=log(sum_freq)

        candidate=[]
        new_sentencepieces=dict()

        for key, val in current_piece.items():
            if freq[key]==0 or not always_keep[key]:
                continue
            elif len(alternatives[key])==0:
                new_sentencepieces[key]=val
            else:
                F= inverted[key]
                F/=vsum
                logprob_sp = log(freq[key])-logsum
                logsum_alt = log(sum_freq+freq[key]*(len(alternatives)-1))

                logprob_alt=0
                for alt in alternatives[key]:
                    logprob_alt += (log(freq[alt]+freq[key])-logsum_alt)

                loss = F*(logprob_sp-logprob_alt)
                candidate.append((key,loss))

        pruned_size = max(len(current_piece)*self.shrinking_rate,self.desired_voc_size)

        for piece,_ in sorted(candidate,key=lambda x:x[1],reverse=True):
            #add piece from candidate in decsengind order of score till piece size reaches to pruned_size
            if len(new_sentencepieces)==pruned_size:
                break
            new_sentencepieces[piece]=current_piece[piece]

        print("prune step {} pieces are pruned".format(len(current_piece) - len(new_sentencepieces)))
        return new_sentencepieces



    def finalize_sentencepiece(self):
        """最終的な処理
        fileへの書き込みをする
        """
        print("finally, {} pieces".format(self.SentencePiece.get_piece_size()))
        piece = self.SentencePiece.get_pieces()
        with open(self.out_voc_file,"w") as f:
            for key,val in sorted(piece.items(),key=lambda x:-x[1]):
                f.write("{}\t{}\n".format(key,val))
        print("written voc to {}".format(self.out_voc_file))

    def build_trie(self,pieces):
        """ building Trie from piece
        """
        Trie = pygtrie.Trie()
        for (key,score) in pieces.items():
            Trie[key]=(key,score)
        self.Trie=Trie
        

    def train(self):
        """ training 
        """
        self.load_sentence()
        seed_sentencepieces = self.make_seed_sentence_piece()
        self.set_sentnece_piece(seed_sentencepieces)


        self.SentencePiece.print_piece()

        for _ in range(3):
            for itr in range(2):#EM iteration loop
                expected,objective,num_tokens = self.run_e_step()
                new_sentencepieces = self.run_m_step(expected)

                self.set_sentnece_piece(new_sentencepieces)
                print("EM sub_iter= {} size={} obj={} num_tokens= {} num_tokens/piece= {}".format(itr,self.SentencePiece.get_piece_size(),objective,num_tokens,num_tokens/self.SentencePiece.get_piece_size()))

            new_sentencepieces=self.prune_piece()
            self.set_sentnece_piece(new_sentencepieces)

        #Save to file
        self.finalize_sentencepiece()



# sample
if __name__=="__main__":
    dummy_arg={"file":"../test/dummy.en","voc":"../res_voc/dummy.en.voc"}
    #dummy_arg={"src_file":"../test/dummy2.en","src_voc":"../res_voc/dummy2.en.voc"}
    #dummy_arg={"src_file":"../test/dummy3.en","tgt_file":None}
    #dummy_arg={"src_file":"../test/dummy.jap"}
    #dummy_arg={"src_file":"../test/dummy4.en"}
    U = UnigramModel(dummy_arg)
    U.train()
