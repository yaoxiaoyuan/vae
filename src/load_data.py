# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 10:38:59 2020

"""
import random
import numpy as np
import torch


def load_vocab(vocab_path):
    """
    """
    vocab = {}
    for line in open(vocab_path, "rb"):
        line = line.decode("utf-8").strip()
        word, word_id = line.split("\t")
        vocab[word] = int(word_id)
    
    return vocab


class DataLoader():
    """
    """
    def __init__(self, file_path, symbols, word2id, 
                 batch_size, max_len, gpu, word_dropout):
        """
        """
        self.PAD = symbols["<PAD>"]
        self.BOS = symbols["<BOS>"]
        self.EOS = symbols["<EOS>"]
        self.UNK = symbols["<UNK>"]
        self.word2id = word2id
        self.max_len = max_len
        self.gpu = gpu
        self.word_dropout = word_dropout
        
        self.cache = []
        lines = [line.strip().split() 
                    for line in open(file_path, "r", encoding="utf-8")]
        random.shuffle(lines)
        lines.sort(key=lambda x:len(x))
        
        cur_len = len(lines[0])
        tmp = []
        for line in lines:
            text = [self.word2id.get(w, self.UNK) for w in line]
            
            if len(text) == cur_len:
                text = [self.BOS] + text + [self.EOS]
                tmp.append(text)
                if len(tmp) == batch_size:
                    self.cache.append(tmp)
                    tmp = []
            else:
                if len(tmp) > 0:
                    self.cache.append(tmp)
                cur_len = len(text)
                text = [self.BOS] + text + [self.EOS]
                tmp = [text]
                
        if len(tmp) > 0:
            self.cache.append(tmp)

    
    def __call__(self):
        """
        """
        random.shuffle(self.cache)

        def process_batch(batch_data):
            """
            """
            max_len = max(len(ss) for ss in batch_data) - 1
            seq = np.zeros([len(batch_data), max_len]) + self.PAD
            target = np.zeros([len(batch_data), max_len]) + self.PAD
            for i,xx in enumerate(batch_data):
                if self.word_dropout > 0:
                    _xx = [ 
                        ww if random.random() > self.word_dropout else self.UNK
                        for ww in xx[:-1]
                    ]
                    seq[i, :len(xx)-1] = _xx
                else:
                    seq[i, :len(xx)-1] = xx[:-1]
                        
                target[i, :len(xx)-1] = xx[1:]
            
            seq = torch.tensor(seq, dtype=torch.long)
            target = torch.tensor(target, dtype=torch.long)
            
            if self.gpu >= 0:
                seq = seq.cuda()
                target = target.cuda()
                
            return seq, target

        for batch in self.cache:
            yield process_batch(batch)
            
            
            
                    
                    
                    