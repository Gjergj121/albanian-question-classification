import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math
from transformers import RobertaModel


class Attention(nn.Module):
    def __init__(self, dimensions, attention_type='general'):
        super(Attention, self).__init__()

        if attention_type not in ['dot', 'general']:
            raise ValueError('Invalid attention type selected.')

        self.attention_type = attention_type
        if self.attention_type == 'general':
            self.linear_in = nn.Linear(dimensions, dimensions, bias=False)

        self.linear_out = nn.Linear(dimensions * 2, dimensions, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()


    def forward(self, query, context):
        """
        Args:
            query (:class:`torch.FloatTensor` [batch size, output length, dimensions]): Sequence of
                queries to query the context.
            context (:class:`torch.FloatTensor` [batch size, query length, dimensions]): Data
                overwhich to apply the attention mechanism.

        Returns:
            :class:`tuple` with `output` and `weights`:
            * **output** (:class:`torch.LongTensor` [batch size, output length, dimensions]):
              Tensor containing the attended features.
            * **weights** (:class:`torch.FloatTensor` [batch size, output length, query length]):
              Tensor containing attention weights.
        """
        batch_size, output_len, dimensions = query.size()
        query_len = context.size(1)

        if self.attention_type == "general":
            query = query.reshape(batch_size * output_len, dimensions)
            query = self.linear_in(query)
            query = query.reshape(batch_size, output_len, dimensions)

        attention_scores = torch.bmm(query, context.transpose(1, 2).contiguous())

        # Compute weights across every context sequence
        attention_scores = attention_scores.view(batch_size * output_len, query_len)
        attention_weights = self.softmax(attention_scores)
        attention_weights = attention_weights.view(batch_size, output_len, query_len)

        mix = torch.bmm(attention_weights, context)

        combined = torch.cat((mix, query), dim=2)
        combined = combined.view(batch_size * output_len, 2 * dimensions)

        output = self.linear_out(combined).view(batch_size, output_len, dimensions)
        output = self.tanh(output)

        return output, attention_weights
   
    
class BiLSTM(nn.Module):
    def __init__(self, embedding_layer, hidden_size, output_size, embed_dim=300, dropout=0.1, use_attention=False, bidirectional=True):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(
            embed_dim, hidden_size, 2, bias=True,
            batch_first=False, dropout=dropout, bidirectional=bidirectional)
        
        self.use_attention = use_attention
        self.n = 1
        if bidirectional:
                self.n = 2
                
        if self.use_attention:
            self.attention_layer = Attention(hidden_size * self.n)
        
        self.embedding_layer = embedding_layer
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.proj_linear = nn.Linear(hidden_size * self.n, hidden_size)
        self.output_linear = nn.Linear(hidden_size, output_size)
        
    def forward(self, input):
        # input -> [S, B]
        embeddings = self.embedding_layer(input)
        embeddings = self.dropout(embeddings) 
        
        output, (hidden, query) = self.lstm(embeddings)
        output = output.permute(1, 0, 2) 
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1).unsqueeze(1)
        
        if self.use_attention:
            output, weights = self.attention_layer(hidden, output)
        else:
            output = output.mean(axis=1)
        
        output = self.relu(self.proj_linear(output))
        output = self.dropout(output) 
        output = self.output_linear(output)
        
        return output.squeeze(1)
    

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    

class TransformerModel(nn.Module):

    def __init__(self, embedding_layer, d_model, nhead, d_hid, num_outputs,
                 nlayers, dropout = 0.2):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = embedding_layer
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, num_outputs)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        # self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
        Returns:
            output Tensor of shape [seq_len, batch_size, n_embeddings]
        """
        lengths = (src != 0).sum(axis=0)
        src_padding_mask = (src == 0)
        
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        encoding_output = self.transformer_encoder(src)
        
        encoding_output[src_padding_mask] = 0
        output = torch.sum(encoding_output, axis=0) / lengths.unsqueeze(1)
        
        output = self.decoder(output)
        return output
    

class BertClassifier(nn.Module):
    def __init__(self, num_outputs, dim=768, roberta='macedonizer/al-roberta-base'):
        super().__init__()
        self.bert = RobertaModel.from_pretrained(roberta)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.linear_1 = nn.Linear(dim, dim//2)
        self.linear_2 = nn.Linear(dim//2, num_outputs)
    
    def forward(self, input):
        bert_output = self.bert(**input)
        output = self.linear_1(self.dropout(bert_output.pooler_output))
        output = self.relu(output)
        output = self.linear_2(self.dropout(output))
        return output
    
    def size(self):
        return sum(p.numel() for p in self.parameters())
    
    

    
    
    
    
    
"""Ky eshte modeli me sqarime dhe komente
"""
class ExlainedBiLSTM(nn.Module):
    def __init__(self, embedding_layer, hidden_size, output_size, embed_dim=300, dropout=0.1, use_attention=False, bidirectional=True):
        super(ExlainedBiLSTM, self).__init__()
        self.lstm = nn.LSTM(
            embed_dim, hidden_size, 2, bias=True,
            batch_first=False, dropout=dropout, bidirectional=bidirectional)
        
        self.use_attention = use_attention
        self.n = 1
        if bidirectional:
                self.n = 2
                
        if self.use_attention:
            self.attention_layer = Attention(hidden_size * self.n)
        
        self.embedding_layer = embedding_layer
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.proj_linear = nn.Linear(hidden_size * self.n, hidden_size)
        self.output_linear = nn.Linear(hidden_size, output_size)
        
    def forward(self, input):
        # input -> [S, B]
        embeddings = self.embedding_layer(input)
        embeddings = self.dropout(embeddings) # [S, B, D]
        embeddings = self.dropout(embeddings)
        # LSTM -> batch_first = False
        output, (hidden, query) = self.lstm(embeddings) # output -> [S, B, N * D], hidden, query -> [L * N, B, D] ku L = numri i LSTM layers, dhe N = 2 nqs eshte bidirectional
        # Kemi dy layers LSTM dhe eshte bidirectional, kjo dmthn qe per cdo layer do kemi dy last hidden vectors (for both directions)
        # Pra hidden shape eshte -> [2 * 2, B, D]
        # dy elementet e pare i perkasin layerit te pare, pastaj dy te fundit te layerit te fundit. 
        # Me poshte marrim dy hidden state te layerit te fundit (sepse ai na intereson), per te dyja drejtimet dhe i bashkojme
        # Pse i bashkojme? Sepse kur eshte bidirectional, output do ket size ne fund (N * D) sepse jane bashkuar te dyja drejtimet bashke, kurse hidden layers e kane vetem D
        # Keshtu qe i bashkojme dhe hidden layers qe te kemi dimension e njejte ne fund. 
        # Pra, po bashkojme dy vektor me dimension [B, D], per te bere dimension [B, 2 * D], dhe me pas i bejme unsqueeze tek dimensioni i dyte per ta sjell shape [B, 1, 2 * D] (shif poshte pse)
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1).unsqueeze(1) 
        
        output = output.permute(1, 0, 2) # e cojme batching ne fillim [B, S, D] sepse do na duhet per attentionin dhe linear layers
        if self.use_attention:
            # Per attention, na duhet final hidden state dhe outputi. 
            # Te dyja na duhet batch i pari, dhe i kemi sjell ne formen hidden [B, 1, N*D] dhe output [B, S, N*D]
            output, weights = self.attention_layer(hidden, output) #  outputi do kete shape [B, N * D]

        # Me poshte thjesht dy linear layer (relu eshte aktivation function, kurse dropout regularizer)
        # Ne fillim i bejme project down dimension. dhe pastaj e cojme tek madhesia e outputit
        output = self.relu(self.proj_linear(output))
        output = self.dropout(output) 
        output = self.output_linear(output)
        
        return output.squeeze(1)