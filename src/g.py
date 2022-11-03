# -*- coding: utf8 -*-
#
from transformers import AutoTokenizer, AutoModel, PreTrainedTokenizerBase
# BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained('hfl/chinese-electra-180g-base-discriminator')
ptm_model = AutoModel.from_pretrained('hfl/chinese-electra-180g-base-discriminator')
