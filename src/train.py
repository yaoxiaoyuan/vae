# -*- coding: utf-8 -*-
"""
Created on Tue May 12 12:05:22 2020

"""
import os
import random
import torch
from torch import optim
from vae import VAE
from load_data import load_vocab, DataLoader
from utils import loss_fn, eval_fn, calc_au, calc_iw_nll, calc_mi

symbols = {"<PAD>":0, "<BOS>":1, "<EOS>":2, "<UNK>":3}

def train(args, logger):
    """
    """
    logger.info(args)
    
    torch.random.manual_seed(args.seed)
    random.seed(args.seed)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    word2id = load_vocab(args.vocab_file)
    train_data_loader = DataLoader(args.train_file, symbols, word2id, 
                                   args.batch_size, args.max_len, 
                                   args.gpu, args.word_dropout)
    val_data_loader = DataLoader(args.val_file, symbols, word2id, 
                                 args.batch_size, args.max_len, 
                                 args.gpu, args.word_dropout)
    test_data_loader = DataLoader(args.test_file, symbols, word2id, 
                                  args.batch_size, args.max_len, 
                                  args.gpu, args.word_dropout)
    
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
    
    optimizer = optim.Adam(vae.parameters(), args.lr)
    train_data_len = len(train_data_loader.cache)
        
    cur_lr = args.lr
    decay_cnt = 0
    steps = 1

    beta = 0.1
    anneal_rate = 0.9 / (max(1, args.warmup * train_data_len))

    for epoch in range(args.num_epoch):
        vae.train()    
        for seq,target in train_data_loader():
            beta = min(1, 0.1 + 0.9 * steps * anneal_rate)
            
            pred,mu,logvar = vae(seq)
            rec, kld = loss_fn(pred, mu, logvar, target)
            
            num_sents = seq.size(0)
            rec = rec / num_sents
            
            loss = rec + beta * kld
            
            if steps % args.print_every == 0:
                logger.info("%d epoch %d steps beta:%.2f lr:%.5f rec: %.2f kld: %.2f" %
                            (epoch, steps % train_data_len, 
                             beta, cur_lr,
                             rec.item(), kld.item()))
            
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(vae.parameters(), args.grad_clip)
            
            optimizer.step()
            optimizer.zero_grad()
                    
            steps += 1
        
        val_rec,val_kld,val_loss,val_elbo,val_ppl = \
            eval_fn(vae, val_data_loader)
        logger.info("%d epoch val rec:%.2f kld:%.2f elbo:%.2f ppl:%.2f" % 
                    (epoch, val_rec, val_kld, val_elbo, val_ppl))
            
        test_rec,test_kld,test_loss,test_elbo,test_ppl = \
            eval_fn(vae, test_data_loader)
        logger.info("%d epoch test rec:%.2f kld:%.2f elbo:%.2f ppl:%.2f" % 
                    (epoch, test_rec, test_kld, test_elbo, test_ppl))

        if epoch == 0:
            best_val_loss = val_loss
            torch.save(vae.state_dict(), args.save_model)
        elif val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(vae.state_dict(), args.save_model)
        elif args.lr_decay > 0:
            decay_cnt += 1
            cur_lr *= args.lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] = cur_lr   
        
        if decay_cnt > args.max_decay:
            break
    

def eval_model(args, logger):
    """
    """
    logger.info(args)
    
    random.seed(args.seed)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    word2id = load_vocab(args.vocab_file)
    test_data_loader = DataLoader(args.test_file, symbols, word2id, 
                                  args.batch_size, args.max_len, 
                                  args.gpu, args.word_dropout)
    
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

    rec,kld,loss,elbo,ppl = eval_fn(vae, test_data_loader)
    au = calc_au(vae, test_data_loader) 
    iw_nll, iw_ppl = calc_iw_nll(vae, test_data_loader)
    mi = calc_mi(vae, test_data_loader)    

    logger.info(
            "test rec:%.2f kld:%.2f elbo:%.2f ppl:%.2f au:%.2f iw_nll:%.2f iw_ppl:%.2f mi:%.2f" % 
            (rec, kld, elbo, ppl, au, iw_nll, iw_ppl, mi))
    