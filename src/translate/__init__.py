# -*- coding: utf-8 -*-
# Natural Language Toolkit: Machine Translation
#
# Copyright (C) 2001-2020 NLTK Project
# Author: Steven Bird <stevenbird1@gmail.com>, Tah Wei Hoon <hoon.tw@gmail.com>
# URL: <http://nltk.org/>
# For license information, see LICENSE.TXT

"""
Experimental features for machine translation.
These interfaces are prone to change.
"""

from translate.api import AlignedSent, Alignment, PhraseTable
from translate.ibm_model import IBMModel
from translate.ibm1 import IBMModel1
from translate.ibm2 import IBMModel2
from translate.ibm3 import IBMModel3
from translate.ibm4 import IBMModel4
from translate.ibm5 import IBMModel5
from translate.bleu_score import sentence_bleu as bleu
from translate.ribes_score import sentence_ribes as ribes
from translate.meteor_score import meteor_score as meteor
from translate.metrics import alignment_error_rate
from translate.stack_decoder import StackDecoder
