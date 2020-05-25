class SentencePiece:
    """ pieceをなんやかんやする
    """

    def __init_s(self):
        """
        sentencepieces: dict[key:peice_surface,val:piece_score]
        """
        self.sentencepieces=[]

    def set_sentence_piece(self,pieces):
        self.sentencepieces=pieces


    def get_pieces(self):
        return self.sentencepieces

    def get_piece_size(self):
        return len(self.sentencepieces)

    def print_piece(self):
        print("current piece: size={}".format(len(self.sentencepieces)))
