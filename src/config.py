# -*- coding: utf-8 -*-
"""
Created on Tue May 12 12:03:44 2020

"""
import argparse

def parse_ptb_args(arguments):
    """
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_file', default='../data/ptb/train.txt')
    parser.add_argument('--val_file', default='../data/ptb/val.txt')
    parser.add_argument('--test_file', default='../data/ptb/test.txt')
    parser.add_argument('--vocab_file', default='../data/ptb/vocab.txt')
    

    parser.add_argument('--cell_type', default='lstm', type=str, 
                        choices = ['lstm', 'gru'])
    parser.add_argument('--vocab_size', default=10002, type=int)
    parser.add_argument('--latent_size', default=32, type=int)
    parser.add_argument('--enc_embedding_size', default=256, type=int)
    parser.add_argument('--enc_hidden_size', default=256, type=int)
    parser.add_argument('--num_enc_layers', default=1, type=int)
    parser.add_argument('--dec_embedding_size', default=256, type=int)
    parser.add_argument('--dec_hidden_size', default=256, type=int)
    parser.add_argument('--num_dec_layers', default=1, type=int)
    parser.add_argument('--bidirectional', action='store_true')
    parser.add_argument('--dropout', default=0.5, type=float)
    
    parser.add_argument('--num_epoch', default=40, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--lr_decay', default=0.5, type=float)
    parser.add_argument('--max_decay', default=5, type=float)
    parser.add_argument('--grad_clip', default=5, type=float)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--warmup', default=0, type=int)
    parser.add_argument('--word_dropout', default=0, type=float)
    parser.add_argument('--save_model', default='../model/ptb')
    parser.add_argument('--max_len', default=82, type=int)
    parser.add_argument('--gpu', default=-1, type=int)
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--print_every', default=100, type=int)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--sample', action='store_true')
    
    args = parser.parse_args(arguments)
    
    return args


def parse_yahoo_args(arguments):
    """
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_file', default='../data/yahoo/train.txt')
    parser.add_argument('--val_file', default='../data/yahoo/val.txt')
    parser.add_argument('--test_file', default='../data/yahoo/test.txt')
    parser.add_argument('--vocab_file', default='../data/yahoo/vocab.txt')
    

    parser.add_argument('--cell_type', default='lstm', type=str, 
                        choices = ['lstm', 'gru'])
    parser.add_argument('--vocab_size', default=20000, type=int)
    parser.add_argument('--latent_size', default=32, type=int)
    parser.add_argument('--enc_embedding_size', default=512, type=int)
    parser.add_argument('--enc_hidden_size', default=1024, type=int)
    parser.add_argument('--num_enc_layers', default=1, type=int)
    parser.add_argument('--dec_embedding_size', default=512, type=int)
    parser.add_argument('--dec_hidden_size', default=1024, type=int)
    parser.add_argument('--num_dec_layers', default=1, type=int)
    parser.add_argument('--bidirectional', action='store_true')
    parser.add_argument('--dropout', default=0.5, type=float)
    
    parser.add_argument('--num_epoch', default=40, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--lr_decay', default=0.5, type=float)
    parser.add_argument('--max_decay', default=5, type=float)
    parser.add_argument('--grad_clip', default=5, type=float)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--warmup', default=0, type=int)
    parser.add_argument('--word_dropout', default=0, type=float)
    parser.add_argument('--save_model', default='../model/yahoo')
    parser.add_argument('--max_len', default=202, type=int)
    parser.add_argument('--gpu', default=-1, type=int)
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--print_every', default=100, type=int)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--sample', action='store_true')
    
    args = parser.parse_args(arguments)
    return args


def parse_yelp_args(arguments):
    """
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_file', default='../data/yelp/yelp.train.txt')
    parser.add_argument('--val_file', default='../data/yelp/yelp.valid.txt')
    parser.add_argument('--test_file', default='../data/yelp/yelp.test.txt')
    parser.add_argument('--vocab_file', default='../data/yelp/vocab.txt')
    

    parser.add_argument('--cell_type', default='lstm', type=str, 
                        choices = ['lstm', 'gru'])
    parser.add_argument('--vocab_size', default=20000, type=int)
    parser.add_argument('--latent_size', default=32, type=int)
    parser.add_argument('--enc_embedding_size', default=512, type=int)
    parser.add_argument('--enc_hidden_size', default=1024, type=int)
    parser.add_argument('--num_enc_layers', default=1, type=int)
    parser.add_argument('--dec_embedding_size', default=512, type=int)
    parser.add_argument('--dec_hidden_size', default=1024, type=int)
    parser.add_argument('--num_dec_layers', default=1, type=int)
    parser.add_argument('--bidirectional', action='store_true')
    parser.add_argument('--dropout', default=0.5, type=float)
    
    parser.add_argument('--num_epoch', default=40, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--lr_decay', default=0.5, type=float)
    parser.add_argument('--max_decay', default=5, type=float)
    parser.add_argument('--grad_clip', default=5, type=float)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--warmup', default=0, type=int)
    parser.add_argument('--word_dropout', default=0, type=float)
    parser.add_argument('--save_model', default='../model/yelp')
    parser.add_argument('--max_len', default=200, type=int)
    parser.add_argument('--gpu', default=-1, type=int)
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--print_every', default=100, type=int)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--sample', action='store_true')
    
    args = parser.parse_args(arguments)

    return args
