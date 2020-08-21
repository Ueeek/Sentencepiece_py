DST="../../align_tokenize_deen/vocs/"
SRC="../../data/wmtEnDe/"

spm_train --vocab_size=8000 --input ${SRC}train.en --model_prefix ${DST}org.en
spm_train --vocab_size=8000 --input ${SRC}train.de --model_prefix ${DST}org.de

