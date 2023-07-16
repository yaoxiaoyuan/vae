# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 14:29:48 2020

"""
import torch
import torch.nn as nn

class Encoder(nn.Module):
    """
    """
    def __init__(self, vocab_size, embedding_size, hidden_size, latent_size, 
                 num_layers, cell_type="lstm", bidirectional=False):
        """
        """
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cell_type = cell_type
        bidirectional = bool(bidirectional)
        self.bidirectional = bidirectional
        
        rnn_output_size = hidden_size
        if bidirectional == True:
            rnn_output_size *= 2
        self.hidden2mulogvar = nn.Linear(rnn_output_size, 2 * latent_size)
        
        if cell_type == "lstm":
            self.rnn = nn.LSTM(input_size=embedding_size,
                               hidden_size=hidden_size,
                               num_layers=num_layers,
                               bidirectional=bidirectional,
                               batch_first=True)
        elif cell_type == "gru":
            self.rnn = nn.GRU(input_size=embedding_size,
                              hidden_size=hidden_size,
                              num_layers=num_layers,
                              bidirectional=bidirectional,
                              batch_first=True)
        
    
    def sample_z(self, mu, logvar):
        """
        """
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
        

    def encode(self, x):
        """
        """
        embeded = self.embedding(x)
        
        h = self.rnn(embeded)[0][:, -1, :]
         
        mu,logvar = torch.chunk(self.hidden2mulogvar(h), 2, -1)

        return mu,logvar

    
    def forward(self, x):
        """
        """
        mu,logvar = self.encode(x)
        
        z = self.sample_z(mu, logvar)
        
        return mu,logvar,z


class Decoder(nn.Module):
    """
    """
    def __init__(self, vocab_size, embedding_size, hidden_size, latent_size, 
                 num_layers, cell_type="lstm", dropout=0, gamma=1):
        """
        """
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.num_layers = num_layers
        self.cell_type = cell_type
        self.gamma = gamma
        
        if cell_type == "lstm":
            self.rnn = nn.LSTM(input_size=embedding_size,
                               hidden_size=hidden_size,
                               num_layers=num_layers,
                               batch_first=True)
        elif cell_type == "gru":
            self.rnn = nn.GRU(input_size=embedding_size,
                              hidden_size=hidden_size,
                              num_layers=num_layers,
                              batch_first=True)
            
        self.dropout = nn.Dropout(dropout)
        self.z2h = nn.Linear(latent_size, hidden_size)
        self.h2logits = nn.Linear(hidden_size, vocab_size)
        self.z2logits = nn.Linear(latent_size, vocab_size)
        

    def forward(self, x, z):
        """
        """
        batch_size = x.size(0)
        
        embeded = self.embedding(x)
        
        embeded = self.dropout(embeded)
        
        _z = z.unsqueeze(1).expand([batch_size, x.size(1), self.latent_size])
        
        dec_input = embeded
        _z_h = self.z2h(z)
        if self.cell_type == "lstm":
            init_h = torch.zeros(self.num_layers, batch_size, self.hidden_size)
            init_c = torch.zeros(self.num_layers, batch_size, self.hidden_size)
            if x.is_cuda:
                init_h = init_h.cuda()
                init_c = init_c.cuda()
            init_h[-1] = _z_h
            h,_ = self.rnn(dec_input, (init_h,init_c))
        elif self.cell_type == "gru":
            init_h = torch.zeros(self.num_layers, batch_size, self.hidden_size)
            if x.is_cuda:
                init_h = init_h.cuda()
            init_h[-1] = _z_h
            h,_ = self.rnn(dec_input, init_h)
        
        h = self.dropout(h)
        
        logits = self.h2logits(h) + self.z2logits(_z)
            
        pred = torch.log_softmax(logits, -1)
        
        return pred


class CNNDecoder(nn.Module):
    """
    """
    def __init__(self, vocab_size, embedding_size, internal_size, 
                 external_size, latent_size, dropout=0, gamma=1):
        """
        """
        super(CNNDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        
        self.vocab_size = vocab_size
        self.internal_size = internal_size
        self.external_size = external_size
        self.latent_size = latent_size
        self.gamma = gamma
        
        self.conv1d_1 = nn.Conv1d(embedding_size + latent_size,
                                  internal_size,
                                  3)
        self.conv1d_2 = nn.Conv1d(internal_size,
                                  internal_size,
                                  3)
        self.conv1d_2 = nn.Conv1d(internal_size,
                                  external_size,
                                  3)
        
        self.dropout = nn.Dropout(dropout)
        self.z2h = nn.Linear(latent_size, hidden_size)
        self.h2logits = nn.Linear(hidden_size, vocab_size)
        self.z2logits = nn.Linear(latent_size, vocab_size)
        

    def forward(self, x, z):
        """
        """
        batch_size = x.size(0)
        
        embeded = self.embedding(x)
        
        embeded = self.dropout(embeded)
        
        _z = z.unsqueeze(1).expand([batch_size, x.size(1), self.latent_size])
        
        dec_input = embeded
        _z_h = self.z2h(z)
        if self.cell_type == "lstm":
            init_h = torch.zeros(self.num_layers, batch_size, self.hidden_size)
            init_c = torch.zeros(self.num_layers, batch_size, self.hidden_size)
            if x.is_cuda:
                init_h = init_h.cuda()
                init_c = init_c.cuda()
            init_h[-1] = _z_h
            h,_ = self.rnn(dec_input, (init_h,init_c))
        elif self.cell_type == "gru":
            init_h = torch.zeros(self.num_layers, batch_size, self.hidden_size)
            if x.is_cuda:
                init_h = init_h.cuda()
            init_h[-1] = _z_h
            h,_ = self.rnn(dec_input, init_h)
        
        h = self.dropout(h)
        
        logits = self.h2logits(h) + self.z2logits(_z)
            
        pred = torch.log_softmax(logits, -1)
        
        return pred


class VAE(nn.Module):
    """
    """
    def __init__(self, symbols, vocab_size, 
                 enc_embedding_size, enc_hidden_size, num_enc_layers, 
                 dec_embedding_size, dec_hidden_size, num_dec_layers, 
                 latent_size, cell_type="lstm", bidirectional=False, 
                 dropout=0, dynamic=True, gamma=0):
        """
        """
        super(VAE, self).__init__()
        self.PAD = symbols["<PAD>"]
        self.BOS = symbols["<BOS>"]
        self.EOS = symbols["<EOS>"]
        self.UNK = symbols["<UNK>"]
        self.encoder = Encoder(vocab_size, enc_embedding_size, enc_hidden_size, 
                               latent_size, num_enc_layers, cell_type, 
                               bidirectional)
        self.decoder = Decoder(vocab_size, dec_embedding_size, dec_hidden_size, 
                               latent_size, num_dec_layers, cell_type, 
                               dropout, dynamic, gamma)
    
    
    def forward(self, x):
        """
        """
        mu,logvar,z = self.encoder(x)
        pred = self.decoder(x, z)
        
        return pred,mu,logvar
        

    def decode(self, x, max_steps):
        """
        """
        mu,logvar,z = self.encoder(x)
        
        batch_size = x.size(0)
        
        hyp = torch.zeros(batch_size, 0, dtype=torch.long)
        y = torch.zeros(batch_size, 1, dtype=torch.long) + self.BOS
        finished = torch.zeros(batch_size, 1, dtype=torch.uint8)
        
        if x.is_cuda:
            hyp = hyp.cuda()
            y = y.cuda()
            finished = finished.cuda()

        _z_h = self.decoder.z2h(z)
        if self.decoder.cell_type == "lstm":
            init_h = torch.zeros(self.decoder.num_layers, batch_size, self.decoder.hidden_size)
            init_c = torch.zeros(self.decoder.num_layers, batch_size, self.decoder.hidden_size)
            if x.is_cuda:
                init_h = init_h.cuda()
                init_c = init_c.cuda()
            init_h[-1] = _z_h
        elif self.decoder.cell_type == "gru":
            init_h = torch.zeros(self.decoder.num_layers, batch_size, self.decoder.hidden_size)
            if x.is_cuda:
                init_h = init_h.cuda()
            init_h[-1] = _z_h
        
        h,c = init_h,init_c
        
        steps = 0
        z_logits = self.decoder.z2logits(z.unsqueeze(1))
        
        while not finished.all() and steps < max_steps: 
            embeded = self.decoder.embedding(y)
            embeded = self.decoder.dropout(embeded)
            dec_input = embeded
        
            if self.decoder.cell_type == "lstm":
                output,(h,c) = self.decoder.rnn(dec_input, (h,c))
            elif self.decoder.cell_type == "gru":
                output,h = self.decoder.rnn(dec_input, h)
        
            output = self.decoder.dropout(output)
        
            logits = self.decoder.h2logits(output) + z_logits
            
            pred = torch.log_softmax(logits, -1)
        
            y = pred.argmax(-1)

            hyp = torch.cat([hyp, y], -1)
            
            steps += 1
            
            finished = (finished | y.eq(self.EOS).byte())
            
            
        return hyp