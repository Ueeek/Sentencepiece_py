class SentencePiece:
    """ manage sentencepieces vocab
    """

    def __init__(self):
        """
        sentencepieces: dict[key:peice_surface,val:piece_score]
        """
        self.sentencepieces = []

    def _set_sentence_piece(self, pieces):
        """
        set argument piece into this state

        Argument:
            pieces(dict): piece[key]=score
        """

        self.sentencepieces = pieces

    def get_pieces(self):
        """
            return self.piece
        """
        return self.sentencepieces

    def get_piece_size(self):
        """
            return len piece
        """
        return len(self.sentencepieces)

    def print_piece(self):
        """
            return len(piece)
        """
        print("current piece: size={}".format(len(self.sentencepieces)))
