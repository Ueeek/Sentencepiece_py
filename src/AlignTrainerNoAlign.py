from AlignmentLossFuncs import alignment_loss

from AlignTrainerBase import AlignTrainerBase
from collections import defaultdict


class AlignTrainerNoAlign(AlignTrainerBase):
    """
    call def train(self,alpha=0.01,back_up_interval=-1,back_up_file=None): to start train
    """
    def __init__(self,arg_src, arg_tgt):
        super().__init__(arg_src,arg_tgt)

    def align_src_func(self, always_keep, alternatives, freq):
        return defaultdict(int)

    def align_tgt_func(self, always_keep, alternatives, freq):
        return defaultdict(int)

    
