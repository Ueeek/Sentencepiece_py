from AlignmentLossFuncs import alignment_loss

from AlignTrainerBase import AlignTrainerBase


class AlignTrainerBoth(AlignTrainerBase):
    """
    call def train(self,alpha=0.01,back_up_interval=-1,back_up_file=None): to start train
    """
    def __init__(self,arg_src, arg_tgt):
        super().__init__(arg_src,arg_tgt)

    def align_src_func(self, always_keep, alternatives, freq):
        return alignment_loss(self.U_src, self.U_tgt, always_keep, alternatives, freq)

    def align_tgt_func(self, always_keep, alternatives, freq):
        return alignment_loss(self.U_tgt, self.U_src, always_keep, alternatives, freq)

