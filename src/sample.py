# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 14:46:43 2020

@author: yaoxiaoyuan
"""
import sys
import os
import random
import torch
from vae import VAE
from load_data import load_vocab, DataLoader

symbols = {"<PAD>":0, "<BOS>":1, "<EOS>":2, "<UNK>":3}

def sample_text(args, logger):
    """
    """
    logger.info(args)
    
    random.seed(args.seed)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    word2id = load_vocab(args.vocab_file)
    id2word = {word2id[w]:w for w in word2id}
    
    vae = VAE(symbols, args.vocab_size, 
              args.enc_embedding_size, args.enc_hidden_size, 
              args.num_enc_layers, args.dec_embedding_size, 
              args.dec_hidden_size, args.num_dec_layers, 
              args.latent_size, args.cell_type, 
              args.bidirectional, args.dropout)
    
    for param in vae.parameters():
        param.data.uniform_(-0.1, 0.1)    
    
    if args.gpu >= 0:
        vae = vae.cuda()

    vae.load_state_dict(torch.load(args.save_model,
                                   map_location=lambda storage, loc: storage))
    
    vae.eval()
    with torch.no_grad():
        for line in sys.stdin:
            text = [word2id.get(w, symbols["<UNK>"]) for w in line.split()]
            text = [[symbols["<BOS>"]] + text]
    
            x = torch.tensor(text, dtype=torch.long)
            
            if args.gpu >= 0:
                x = x.cuda()
            
            rec_x = vae.decode(x, args.max_len)[0].cpu().numpy()
            
            rec_text = " ".join([id2word[w] for w in rec_x])
            
            print(rec_text)
    
        
    
    