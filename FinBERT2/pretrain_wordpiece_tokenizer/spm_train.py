import sentencepiece as spm
#spm.SentencePieceTrainer.train('--input=chinese_corpus.txt --model_prefix=bpe_vocab_5w --model_type=bpe --num_threads=32 --vocab_size=30000 --train_extremely_large_corpus=true')
# spm.SentencePieceTrainer.train('--input=chinese_corpus_all_sample.txt --model_prefix=unigram_all --num_threads=16 --vocab_size=50000 --train_extremely_large_corpus=true')

import sentencepiece as spm
sp = spm.SentencePieceProcessor(model_file='unigram_all.model')

print(sp.encode('This is a test'))
print(sp.EncodeAsPieces('This is a test'))