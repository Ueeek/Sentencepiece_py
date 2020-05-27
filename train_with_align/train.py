import sys
sys.path.append("../src/")
from UnigramModel import UnigramModel

def prune_jointly(U_en,U_ja):
    pass

def Train_En_JA():
    print("Train")
    arg_en={
        "file":"./corpus/test.en",
        "voc":"./res_voc/dummy.en.voc",
        "shrinking_rate":0.75,
        "desired_voc_size":4000,
        "seed_sentence_piece_size":1e5
    }
    arg_ja={
        "file":"./corpus/test.jap",
        "voc":"./res_voc/dummy.jap.voc",
        "shrinking_rate":0.75,
        "desired_voc_size":4000,
        "seed_sentence_piece_size":1e5
    }
    U_en = UnigramModel(arg_en)
    U_ja = UnigramModel(arg_ja)

    #load sentence
    U_en.load_sentence()
    U_ja.load_sentence()
    #seed_piece
    seed_en = U_en.make_seed_sentence_piece()
    U_en.set_sentnece_piece(seed_en)

    seed_ja = U_ja.make_seed_sentence_piece()
    U_ja.set_sentnece_piece(seed_ja)

    #Start EM
    for _ in range(3):
        for itr in range(2):
            #E
            exp_en,obj_en,n_token_en = U_en.run_e_step()
            exp_ja,obj_ja,n_token_ja = U_ja.run_e_step()

            #M
            new_pieces_en = U_en.run_m_step(exp_en)
            new_piece_ja = U_ja.run_m_step(exp_ja)

            # update
            U_en.set_sentnece_piece(new_pieces_en)
            U_ja.set_sentnece_piece(new_piece_ja)
            
            print("EN EM sub_iter= {} size={} obj={} num_tokens= {} num_tokens/piece= {}".format(itr,U_en.SentencePiece.get_piece_size(),obj_en,n_token_en,n_token_en/U_en.SentencePiece.get_piece_size()))
            print("JA EM sub_iter= {} size={} obj={} num_tokens= {} num_tokens/piece= {}".format(itr,U_ja.SentencePiece.get_piece_size(),obj_ja,n_token_ja,n_token_ja/U_ja.SentencePiece.get_piece_size()))
        
        new_piece_en = U_en.prune_piece()
        new_piece_ja = U_ja.prune_piece()

        U_en.set_sentnece_piece(new_piece_en)
        U_ja.set_sentnece_piece(new_piece_ja)

    U_en.finalize_sentencepiece()
    U_ja.set_sentnece_piece()

if __name__=="__main__":
    Train_En_JA()
