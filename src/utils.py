# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 11:33:10 2020

"""
from datetime import datetime
import logging
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions.normal import Normal

def gaussian_kld(mu, logvar):
    """
    """
    kld = -0.5 * torch.sum(logvar - torch.pow(mu, 2) - torch.exp(logvar) + 1, 
                           1)
    return kld


def loss_fn(pred, mu, logvar, target):
    """
    """
    vocab_size = pred.size(-1)
    rec = F.nll_loss(pred.view(-1, vocab_size), 
                     torch.flatten(target), 
                     reduction="sum")
    kld = torch.mean(gaussian_kld(mu, logvar))
    
    return rec, kld


def calc_au(vae, data_loader, delta=0.01):
    """
    """
    vae.eval()
    with torch.no_grad():
        cnt = 0
        for seq,target in data_loader():
            mu, _, __ = vae.encoder(seq)
            if cnt == 0:
                mu_sum = mu.sum(dim=0, keepdim=True)
            else:
                mu_sum = mu_sum + mu.sum(dim=0, keepdim=True)
            cnt += mu.size(0)
        
        mu_mean = mu_sum / cnt
        
        cnt = 0
        for seq,target in data_loader():
            mu, _, __ = vae.encoder(seq)
            if cnt == 0:
                var_sum = ((mu - mu_mean) ** 2).sum(dim=0)
            else:
                var_sum = var_sum + ((mu - mu_mean) ** 2).sum(dim=0)
            cnt += mu.size(0)
        
        au_var = var_sum / (cnt - 1)
        au = (au_var >= delta).sum().item()
    
    return au 


def calc_iw_nll(vae, data_loader, n_samples=512, batch_size=128):
    """
    """
    vae.eval()
    with torch.no_grad():
        num_sents = 0
        num_words = 0
        nll_loss = 0
        for seq,target in data_loader():
            for i in range(seq.size(0)):
                seq_i = seq[i:i+1,:]
                target_i = target[i:i+1,:]
                
                num_sents += 1
                num_words += seq.size(1)
                
                mu,logvar = vae.encoder.encode(seq_i)
                
                seq_i = seq_i.expand([batch_size, seq_i.size(1)])
                target_i = target_i.expand([batch_size, target_i.size(1)])
                mu = mu.expand([batch_size, mu.size(1)])
                logvar = logvar.expand([batch_size, logvar.size(1)])
                std = torch.exp(0.5 * logvar)
                
                prior = Normal(torch.zeros_like(mu), torch.ones_like(std))
                posterior = Normal(mu, std)
                
                tmp = []
                for _ in range(n_samples//batch_size):
                    z = vae.encoder.sample_z(mu, logvar)
                    
                    logpz = prior.log_prob(z).sum(-1)
                    logqz = posterior.log_prob(z).sum(-1)
                    
                    pred = vae.decoder(seq_i, z)
                    pred = pred.view(-1, pred.size(-1))
                    logpxz = -F.nll_loss(pred, 
                                         target_i.flatten(), 
                                         reduction="none")
                    logpxz = torch.sum(logpxz.view(-1, seq_i.size(1)), -1)
                    
                    tmp.append(logpxz + logpz - logqz)
                
                logp = torch.logsumexp(torch.cat(tmp), dim=-1) - np.log(n_samples)
                nll_loss -= logp.item()
            
        nll = nll_loss / num_sents
        ppl = np.exp(nll * num_sents / num_words)
            
    return nll,ppl
        

def calc_mi(vae, data_loader):
    """
    """
    mi = 0
    num_sents = 0
    vae.eval()
    with torch.no_grad():
        for seq,target in data_loader():
            mu,logvar = vae.encoder.encode(seq)
            batch_size,latent_size = mu.size()

            post_dist = Normal(mu, torch.exp(0.5 * logvar))
            neg_entropy = -post_dist.entropy().sum(-1).mean()

            z = vae.encoder.sample_z(mu, logvar)
            
            mu = mu.repeat(batch_size, 1)
            logvar = logvar.repeat(batch_size, 1)
            z = z.unsqueeze(1).repeat(1,batch_size,1).view(-1, latent_size)
            post_dist = Normal(mu, torch.exp(0.5 * logvar))
            
            log_density = torch.sum(post_dist.log_prob(z), -1).view(batch_size, batch_size)
            
            log_qz = torch.logsumexp(log_density, dim=1) - math.log(batch_size)
            
            _mi = (neg_entropy - log_qz.mean(-1)).item() 
            
            mi += (_mi * batch_size) 
            num_sents += batch_size
        
    return mi / num_sents
            

def eval_fn(vae, data_loader):
    """
    """
    vae.eval()
    total_rec = 0
    total_kl = 0
    total_sents = 0
    total_words = 0
    with torch.no_grad():
        for seq,target in data_loader():
            
            pred,mu,logvar = vae(seq)[:3]
            rec, kld = loss_fn(pred, mu, logvar, target)
            
            num_sents = seq.size(0)
            num_words = seq.size(0) * seq.size(1)
            rec = rec / num_sents

            total_rec += (rec.item() * num_sents)
            total_kl += (kld.item() * num_sents)
            total_sents += num_sents
            total_words += num_words
            
        rec = total_rec / total_sents
        kld = total_kl / total_sents
    
        loss = rec + kld
        elbo = -loss
        ppl = np.exp((total_rec + total_kl) / total_words)
            
    return rec, kld, loss, elbo, ppl


def build_logger():
    """
    """
    filename = datetime.today().strftime('../logger/%Y-%m-%d-%H-%M-%S.log')
    format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(filename = filename, 
                        level = logging.INFO, 
                        format = format_str)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter(format_str)
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    logger = logging.getLogger(__name__)
    
    return logger


