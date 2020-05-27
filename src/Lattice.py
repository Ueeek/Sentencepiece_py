from collections import defaultdict
import heapq
from util import LogSumExp
from math import exp
import pygtrie

class Node:
    """ Node of lattice
    """
    def __init__(self):
        """ Attributes
                piece: surace of piece
                pos: begin pos of this node in the sentence
                length: length of this node(this piece)
                is_vocav: True if this is piece else  not-vocab(eos,bos)
                score: unigram score of this node
                backtrace_score: backtrace_score of Viterbi
                prev: prev node of Viterbi path
        """
        self.piece = None
        self.pos = None
        self.length=None
        self.node_id= None
        self.is_vocab=None
        self.score = None
        self.backtrace_score= 0 
        self.prev= None

    #def set_piece(self,piece): self.piece = piece
    #def set_pos(self,pos): self.pos=pos
    #def set_length(self,length): self.length = length
    #def set_node_id(self,node_id): self.node_id=node_id
    #def set_score(self,score):self.score = score
    #def set_backtrace_score(self,score):self.backtrace_score=score
    #def set_prev(self,prev):self.prev=prev

    def __str__(self):
        return "Node piece:{} id:{} pos:{} score:{}".format(self.piece,self.node_id,self.pos,self.score)





class Lattice:
    """ Tokenization Lattice of the sentence
    """
    def __init__(self):
        """ Attributes
            nodes(dict): nodes of this Lattice. key is node id.
            surface(str): tokenizing sentence
            surfaces(list): all suffixes .list[i]=> suffixes that begin at pos i
            size(int): len(surfaces)
            begin_nodes(list):begin_nodes[i]=> node-ids begin at pos[i]
            end_nodes(list):end_nodes[i]=> node-ids that end at pos[i]
        """

        self.nodes=dict()
        self.surface=None
        self.surfaces=None
        self.size=None
        self.begin_nodes_=None
        self.end_nodes_=None


    def get_bos_nodes(self)->int:
        """ return bos node id
        """
        return self.end_nodes_[0][0]

    def get_eos_nodes(self)->int:
        """ return eos node id
        """
        return self.begin_nodes_[self.size][0]

    
    def set_sentence(self,s:str):
        """ set sentence into Lattice
        """

        self.surface=s
        surfaces=[s[i:] for i in range(len(s))] #all suffixes
        self.surfaces=surfaces
        
        self.size = len(surfaces)

        self.begin_nodes_=[[] for _ in range(self.size+1)]
        self.end_nodes_=[[] for _ in range(self.size+1)]

        #set bos node to this Lattice
        bos = Node()
        bos.is_vocab=False
        bos.pos = 0
        bos.node_id = len(self.nodes)
        bos.piece=("_bos")
        bos.score=0

        self.end_nodes_[0].append(bos.node_id)
        self.nodes[bos.node_id]=bos
        
        #set eos node to this Lattice
        eos = Node()
        eos.node_id=len(self.nodes)
        eos.is_vocab=False
        eos.pos=self.size
        eos.piece="_eos_"
        eos.score=0

        self.begin_nodes_[self.size].append(eos.node_id)
        self.nodes[eos.node_id]=eos


    def insert_node(self,pos:int,piece:str,is_vocab:bool,score:int):
        """ insert node into Lattice
        """

        #init node
        node = Node()
        node.pos=pos
        node.piece=piece
        node.length=len(piece)
        node.node_id=len(self.nodes)
        node.score=score
        node.is_vocab= is_vocab

        # set to Lattice
        self.nodes[node.node_id]=node
        self.begin_nodes_[pos].append(node.node_id)
        self.end_nodes_[pos+node.length].append(node.node_id)

    def populate_nodes(self,pieces:dict,Trie):
        """ make Lattice of the sentence with current sentence pieces
        Args:
            pieces: sentence piece dict[piece]=score
            Trie: Trie data structure to find common prefixes efficiently
        """

        #find pieces that have common prefix with surfaces[begin_pos]
        for begin_pos in range(self.size):
            common_prefixes_trie = [v[1] for v in Trie.prefixes(self.surfaces[begin_pos])]
            for (key,score) in common_prefixes_trie:
                self.insert_node(begin_pos,key,True,score)

            if all(len(v[0])>1 for v in common_prefixes_trie):#not contain single char in the common_prefixes
                min_score=min(val for _,val in pieces.items())
                #TODO UNK IDの処理ができてない 怪しめ
                self.insert_node(begin_pos,self.surfaces[begin_pos][0],False,min_score-10)

    def populate_marginal(self,freq:int)->(int,dict):
        """ calculate Marginal Probability
        Args:
            freq: frequenct of the string

        Return:
            expected(dict): expected of each piece ,dict[piece]=score
            Z: objective?
            
        """

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

        expected=defaultdict(int)
        Z = forward_accm[self.begin_nodes_[self.size][0]]
        for pos in range(self.size):
            for node in self.begin_nodes_[pos]:
                piece = self.nodes[node].piece
                if not self.nodes[node].is_vocab:#node is eos or bos
                    continue #id
                expected[piece]+= freq*exp(forward_accm[node]+self.nodes[node].score+backward_accm[node]-Z)

        return Z*freq,expected

    def Viterbi(self):
        """ calculate Viterbi path
        Return
            best_path_ids(list): Viterbi path. list of node_id consists of best tokenization of the sentence
        """
        for pos in range(self.size+1):
            for rnode in self.begin_nodes_[pos]:
                self.nodes[rnode].prev=None
                best_score=0
                best_node_id=None

                for lnode in self.end_nodes_[pos]:
                    score=self.nodes[lnode].backtrace_score+self.nodes[rnode].score
                    if best_node_id is None or score>best_score:
                        best_node_id=lnode
                        best_score = score

                assert best_node_id is not None ,"faild to faind best path in Viterbi surface:{} rnode:{}".format(self.surface,self.nodes[rnode].piece)

                self.nodes[rnode].prev = best_node_id
                self.nodes[rnode].backtrace_score = best_score
                
        # back trace
        best_path_ids=[]
        best_path_surfaces=[]

        cur_node=self.begin_nodes_[self.size][0]
        cur_node=self.nodes[cur_node].prev
        while self.nodes[cur_node].prev is not None:
            best_path_ids.append(self.nodes[cur_node].node_id)
            best_path_surfaces.append(self.nodes[cur_node].piece)
            cur_node = self.nodes[cur_node].prev

        best_path_ids = list(reversed(best_path_ids))
        best_path_surfaces = list(reversed(best_path_surfaces))
        
        assert self.surface=="".join(best_path_surfaces),"surface: {}  viterbi: {}".format(self.surface," ".join(best_path_surfacesf))
        return best_path_ids

    def NBest(self,nbest_size:int)->list:
        """ calculate Nbest tokenization of the sentence
        Return:
            results(list): results[i] = i-th best tokenization ids
        """

        if nbest_size==1:
            return self.Viterbi()

        results=[]
        kPreallocatedHypothesisSize = 512

        #hypothesis
        Agenda=[]
        Hypos=dict()

        #node is tuple (fx,gx,next_hypothesis,node_id,hypo_id)
        def make_hypo(fx:int,gx:int,next_hypothesis:int,node_id:int,hypo_id:int)->tuple:
            """ 
                fx
                gx:
                new_hypothesis:
                node_id: node id of the Lattice (self.node)
                hypo_id: id of hypo
            """
            return (-fx,-gx,{"fx":fx,"gx":gx,"next_hypo":next_hypothesis,"node_id":node_id,"hypo_id":hypo_id})

        #init eos hypo(start from eos)
        eos_node_id=self.get_eos_nodes()
        eos_score = self.nodes[eos_node_id].score
        eos_hypo_id=len(Hypos)

        eos_hypo = make_hypo(eos_score,eos_score,None,eos_node_id,eos_hypo_id)

        Hypos[eos_hypo_id]=eos_hypo

        #add node into Agenda: heapq. acsending order of -fx(descending order of fx)
        heapq.heappush(Agenda,eos_hypo)

        #Run viterbi to fill bachtrace score of each node
        self.Viterbi()

        while Agenda:
            (_,_,top_hypo) = heapq.heappop(Agenda)

            #Reach to BOS
            if top_hypo["node_id"]==self.get_bos_nodes():
                tmp_res=[]

                (_,_,nex)=Hypos[top_hypo["hypo_id"]]
                while nex["next_hypo"] is not None:
                    tmp_res.append(nex["next_hypo"])
                    _,_,nex = Hypos[nex["next_hypo"]]
                results.append(tmp_res)
                if len(results)==nbest_size:
                    break
                continue

            #expands new node ending at node->pos
            for lnode in self.end_nodes_[self.nodes[top_hypo["node_id"]].pos]:
                _,_,cur_hypo = Hypos[top_hypo["hypo_id"]]
                cur_gx =cur_hypo["gx"]
                new_gx= self.nodes[lnode].score + cur_gx
                new_fx = self.nodes[lnode].backtrace_score + cur_gx
                new_next = top_hypo["hypo_id"]

                new_hypo_id=len(Hypos)
                new_hypo=make_hypo(new_fx,new_gx,new_next,lnode,new_hypo_id)

                heapq.heappush(Agenda,new_hypo)
                if new_hypo_id in Hypos.keys():
                    print("kesy")
                Hypos[new_hypo_id]=new_hypo

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

        result_node_ids=[]
        for res in results:
            #resには、hypo_idの列が入ってる。他の何かに変えなければ->node_idで帰るようにする
            tmp=[]
            tmp_node_id=[]
            for h in res[:-1]:
                _,_,cur_hypo=Hypos[h]
                node_id = cur_hypo["node_id"]
                tmp.append(self.nodes[node_id].piece)
                tmp_node_id.append(node_id)
            assert self.surface=="".join(map(str,tmp)),"surface {} tmp:{}".format(self.surface," ".join(tmp))
            result_node_ids.append(tmp_node_id)
        return result_node_ids

    
    def __debug_begin_nodes(self):
        """ function for debug
        """

        print("debug_begin_nodes")
        for i in range(self.size+1):
            print("begin_pos=>",i)
            for be in self.begin_nodes_[i]:#be=node_id
                print("\t node:{}".format(self.nodes[be]))

    def __debug_end_nodes(self):
        """ function for debug
        """

        print("debug_end_nodes")
        for i in range(self.size+1):
            print("end_pos=>",i)
            for be in self.end_nodes_[i]:#be=node_id
                print("\t node:{}".format(self.nodes[be]))

