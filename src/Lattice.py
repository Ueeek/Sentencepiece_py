from collections import defaultdict
import heapq
from util import LogSumExp
from math import exp
class Node:
    def __init__(self):
        self.piece = None
        self.pos = None
        self.length=None
        self.node_id= None
        self.vocab_id= None
        self.score = None
        self.backtrace_score= 0 
        self.prev= None

    def set_piece(self,piece): self.piece = piece
    def set_pos(self,pos): self.pos=pos
    def set_length(self,length): self.length = length
    def set_node_id(self,node_id): self.node_id=node_id
    def set_vocab_id(self,vocab_id):self.vocab_id=vocab_id
    def set_score(self,score):self.score = score
    def set_backtrace_score(self,score):self.backtrace_score=score
    def set_prev(self,prev):self.prev=prev

    def __str__(self):
        return "Node piece:{} id:{} pos:{} score:{}".format(self.piece,self.node_id,self.pos,self.score)





class Lattice:
    def __init__(self):
        self.nodes=dict()
        self.surface=None
        self.surfaces=None
        self.size=None
        self.begin_nodes_=None
        self.end_nodes_=None

    
    def get_begin_nodes(self,pos:int)->list:
        return self.begin_nodes_[pos]

    def get_end_nodes(self,pos:int) ->list:
        return self.end_nodes_[pos]


    def get_bos_nodes(self)->int:
        """ return bos node id
        """
        return self.end_nodes_[0][0]

    def get_eos_nodes(self)->int:
        """ return eos node id
        """
        return self.begin_nodes_[self.size][0]

    
    def set_sentence(self,s:str):
        self.surface=s
        surfaces=[]
        for i in range(len(s)):
            surfaces.append(s[i:])


        self.surfaces=surfaces
        size = len(surfaces)
        self.size=size

        self.begin_nodes_=[[] for _ in range(size+1)]
        self.end_nodes_=[[] for _ in range(size+1)]

        bos = Node()
        bos.set_vocab_id(-1)
        bos.set_pos(0)
        bos.set_node_id(len(self.nodes))
        bos.set_piece("_bos_")
        bos.set_score(0)
        self.end_nodes_[0].append(bos.node_id)
        self.nodes[bos.node_id]=bos
        
        eos = Node()
        eos.set_node_id(len(self.nodes))
        eos.set_vocab_id(-1)
        eos.set_pos(size)
        eos.set_piece("_eos_")
        eos.set_score(0)
        self.begin_nodes_[size].append(eos.node_id)
        self.nodes[eos.node_id]=eos


    def insert_node(self,pos,piece,vocab_id,score):
        node = Node()
        node.set_pos(pos)
        node.set_piece(piece)
        node.set_length(len(piece))
        node.set_node_id(len(self.nodes))
        node.set_vocab_id(vocab_id)
        node.set_score(score)
        #id管理をどうするかも問題
        self.nodes[node.node_id]=node
        self.begin_nodes_[pos].append(node.node_id)
        self.end_nodes_[pos+node.length].append(node.node_id)
        return node

    def populate_nodes(self,pieces):
        """latticeにする
        """

        for begin_pos in range(self.size):
            #surfaces[i]と共通の接頭辞を持つpieceを見つける
            #TODO あとでtrieで高速化する
            #print("surface=>",self.surfaces[begin_pos])
            common_suffixs=[]
            for id,(piece,score) in enumerate(pieces.items()):
                if len(piece)>len(self.surfaces[begin_pos]):continue
                if all(p==s for p,s in zip(piece,self.surfaces[begin_pos])):
                    #print("common_suffixs=>{} score=>{}".format(piece,score))
                    self.insert_node(begin_pos,piece,id,score)
                    common_suffixs.append(piece)

            #UNK の処理。common_suffixsのなかに1文字のものがないならUNK処理する
            if all(len(v)>1 for v in common_suffixs):
            #if len(common_suffixs)==0:
                #print("UNK",common_suffixs)
                min_score=min(val for _,val in pieces.items())
                #TODO scoreは怪しい
                #print("unk_surface=>",self.surfaces[begin_pos])
                #print("beg=>",self.surfaces[begin_pos])
                #多分同じ実装ができていると思う
                self.insert_node(begin_pos,self.surfaces[begin_pos][0],-1,min_score-10)
            #latticeにセットする
        #print(self.surface)
        #self.debug_begin_nodes()
        #self.debug_end_nodes()
        #print("input something to continue")
        #input()
    def populate_marginal(self,freq):
        """ calculate Marginal Probability
        """
        #print("nodes=?>",self.nodes)

        forward_accm=[0]*(len(self.nodes)+1)
        backward_accm=[0]*(len(self.nodes)+1)


        for pos in range(self.size+1):
            for rnode_id in self.begin_nodes_[pos]:
                for lnode_id in self.end_nodes_[pos]:
                    forward_accm[rnode_id]=LogSumExp(forward_accm[rnode_id],self.nodes[lnode_id].score+forward_accm[lnode_id],lnode_id==self.end_nodes_[pos][0])

        for pos in reversed(range(self.size+1)):
            for lnode_id in self.end_nodes_[pos]:
                for rnode_id in self.begin_nodes_[pos]:
                    backward_accm[lnode_id]=LogSumExp(backward_accm[lnode_id],self.nodes[rnode_id].score+backward_accm[rnode_id],rnode_id==self.begin_nodes_[pos][0])

        #print("self=>",self.surface)
        #print("for->",forward_accm)
        #print("back=>",backward_accm)
        

        expected=defaultdict(int)
        Z = forward_accm[self.begin_nodes_[self.size][0]]
        for pos in range(self.size):
            for node in self.begin_nodes_[pos]:
                piece = self.nodes[node].piece
                vocab_id = self.nodes[node].vocab_id
                if vocab_id<0:
                    continue #id=-1でeosとかbosの時
                expected[piece]+= freq*exp(forward_accm[node]+self.nodes[node].score+backward_accm[node]-Z)

        return Z*freq,expected

    def Viterbi(self):
        """ calculate Viterbi path
        """
        for pos in range(self.size+1):
            for rnode in self.begin_nodes_[pos]:
                #print("rnode_suf=>",self.nodes[rnode].piece)
                self.nodes[rnode].prev=None
                best_score=0
                best_node_id=None

                for lnode in self.end_nodes_[pos]:
                    score=self.nodes[lnode].backtrace_score+self.nodes[rnode].score
                    if best_node_id is None or score>best_score:
                        best_node_id=lnode
                        best_score = score

                #ここでこけるのでdebug
                if best_node_id is None:
                    print("surface:=>",self.surface)
                    print("rnode=>",self.nodes[rnode].piece)
                    print("self.surfaces:=>",self.surfaces)
                    self.debug_begin_nodes()
                    self.debug_end_nodes()
                assert best_node_id is not None ,"faild to faind best path in Viterbi"
                self.nodes[rnode].prev = best_node_id
                self.nodes[rnode].backtrace_score = best_score
                
        # back trace
        results=[]
        res_suf=[]

        node=self.begin_nodes_[self.size][0]
        node=self.nodes[node].prev
        while self.nodes[node].prev is not None:
            results.append(self.nodes[node].node_id)
            res_suf.append(self.nodes[node].piece)
            node = self.nodes[node].prev

        results = list(reversed(results))
        res_suf = list(reversed(res_suf))
        
        
        assert self.surface=="".join(res_suf),"surface: {}  viterbi: {}".format(self.surface," ".join(res_suf))
        return results

    def NBest(self,nbest_size:int)->list:
        print("call Nbest with=>",nbest_size)
        if nbest_size==1:
            return self.Viterbi()

        results=[]

        kPreallocatedHypothesisSize = 512

        Agenda=[] #heapq (key is fx)
        Hypos=dict()

        #node is tuple (fx,gx,next_hypothesis,node_id,hypo_id)
        #fxの大きい順に優先度が高くあって欲しい
        eos_node_id=self.get_eos_nodes()
        eos_score = self.nodes[eos_node_id].score
        eos_hypo_id=len(Hypos)
        eos_tup=(-eos_score,-eos_score,None,eos_node_id,eos_hypo_id)
        #eos_node=(eos_score,eos_score,None,eos_node_id)

        Hypos[eos_hypo_id]=eos_tup

        heapq.heappush(Agenda,eos_tup)

        #Run viterbi to fill bachtrace score of each node
        self.Viterbi()

        while Agenda:
            top_tup = heapq.heappop(Agenda)
            (top_fx,top_gx,top_prev,top_id,top_hypo_id) = top_tup

            #Reach to BOS
            if top_id==self.get_bos_nodes():
                print("reach to bos in NBest search")
                tmp_res=[]

                #tupleのindex acssessは見た目的にわかりにくくて好きじゃないので、dictかなんかにしたい。
                nex=Hypos[top_hypo_id]
                while nex[2] is not None:
                    tmp_res.append(nex[2])
                    nex = Hypos[nex[2]]
                results.append(tmp_res)
                if len(results)==nbest_size:
                    break
                continue

            #expands new node ending at node->pos
            #tupleのidがかぶりんちょしそう
            for lnode in self.end_nodes_[self.nodes[top_id].pos]:
                new_gx= self.nodes[lnode].score + -Hypos[top_hypo_id][1]
                new_fx = self.nodes[lnode].backtrace_score + -Hypos[top_hypo_id][1]
                new_next = top_hypo_id
                new_hypo_id=len(Hypos)
                new_hypo_tup=(-new_fx,-new_gx,new_next,lnode,new_hypo_id)

                heapq.heappush(Agenda,new_hypo_tup)
                if new_hypo_id in Hypos.keys():
                    print("kesy")
                Hypos[new_hypo_id]=new_hypo_tup

            #枝かり
            kMaxAgendaSize = 1e5
            kMinAgendaSize = 512

            if len(Agenda)>=kMaxAgendaSize:
                #new_agendaに必要なだけ移し替える
                #TODO ここで、hypoの中身も消すべき?
                print("##Warning##\t Too big agenda. it will be shrinking")
                remove_size = min(kMinAgendaSize,nbest_size*10)
                new_Agenda=[]
                for _ in range(remove_size):
                    t = heapq.heappop(Agenda)
                    heapq.heappush(new_Agenda,t)
                Agenda = new_Agenda

        print("surface=>", self.surface)
        for res in results:
            #resには、hypo_idの列が入ってる。他の何かに変えなければ
            tmp=[]
            for h in res[:-1]:
                node_id = Hypos[h][3]
                tmp.append(self.nodes[node_id].piece)
            print(" ".join(map(str,tmp)))
            assert self.surface=="".join(map(str,tmp)),"surface {} tmp:{}".format(self.surface," ".join(tmp))
        return results







    
    def debug_begin_nodes(self):
        print("debug_begin_nodes")
        for i in range(self.size+1):
            print("begin_pos=>",i)
            for be in self.begin_nodes_[i]:#be=node_id
                print("\t node:{}".format(self.nodes[be]))

    def debug_end_nodes(self):
        print("debug_end_nodes")
        for i in range(self.size+1):
            print("end_pos=>",i)
            for be in self.end_nodes_[i]:#be=node_id
                print("\t node:{}".format(self.nodes[be]))

