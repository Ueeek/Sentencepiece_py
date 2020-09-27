#from AlignmentLossFuncs import alignment_loss

from AlignTrainerBase import AlignTrainerBase


class AlignTrainerBoth(AlignTrainerBase):
    """
    call def train(self,alpha=0.01,back_up_interval=-1,back_up_file=None): to start train
    """
    def __init__(self,arg_src, arg_tgt):
        super().__init__(arg_src,arg_tgt)

    def align_src_func(self, always_keep, alternatives, freq):
        return self.alignment_loss(always_keep, alternatives, freq,src_turn=True)

    def align_tgt_func(self, always_keep, alternatives, freq):
        return self.alignment_loss(always_keep, alternatives, freq,src_turn=False)

    
#    def train(self,alpha=0.01,back_up_interval=-1,back_up_file=None,sample_rate=1.0,em_steps=2):
#        self.sample_rate=sample_rate
#        self.em_steps=em_steps
#        self.alpha=alpha
#        self,back_up_file=back_up_file
#        self.back_up_interval=back_up_interval
#
#        super().train()
#
