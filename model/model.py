import torch
import torch.nn as nn
import torch.nn.functional as F

class BiDAF(nn.Module):
    def __init__(self, args, pretrained):
        super(BiDAF, self).__init__()
        self.args = args

        # 1. Character Embedding Layer
        self.char_emb = nn.Embedding(args.char_vocab_size, args.char_dim, padding_idx=1)
        # issue: char embedding initialization?
        nn.init.uniform_(self.char_emb.weight, -0.05, 0.05)

        self.char_conv = nn.Conv2d(1, args.char_channel_size, (args.char_dim, args.char_channel_width))

        # 2. Word Embedding Layer
        # initialize word embedding with GloVe
        self.word_emb = nn.Embedding.from_pretrained(pretrained, freeze=True)

        # highway network
        assert self.args.hidden_size * 2 == (self.args.char_channel_size + self.args.word_dim)
        for i in range(2):
            setattr(self, f'highway_linear{i}',
                    nn.Sequential(nn.Linear(args.hidden_size * 2, args.hidden_size * 2), nn.ReLU()))
            setattr(self, f'highway_gate{i}',
                    nn.Sequential(nn.Linear(args.hidden_size * 2, args.hidden_size * 2), nn.Sigmoid()))

        # 3. Contextual Embedding Layer
        self.context_LSTM = nn.LSTM(input_size=args.hidden_size * 2,
                                    hidden_size=args.hidden_size,
                                    num_layers=1,
                                    bidirectional=True,
                                    batch_first=True)

        # 4. Attention Flow Layer
        self.att_weight = nn.Linear(args.hidden_size * 6, 1)

        # 5. Modeling Layer
        self.modeling_LSTM = nn.LSTM(input_size=args.hidden_size * 8,
                                     hidden_size=args.hidden_size,
                                     num_layers=2,
                                     bidirectional=True,
                                     batch_first=True)

        # 6. Output Layer
        self.p1_weight = nn.Linear(args.hidden_size * 10, 1)
        self.p2_weight = nn.Linear(args.hidden_size * 10, 1)
        self.output_LSTM = nn.LSTM(input_size=args.hidden_size * 2,
                                   hidden_size=args.hidden_size,
                                   num_layers=1,
                                   bidirectional=True,
                                   batch_first=True)

        self.reset_params()

    def reset_params(self):
        # TODO: careful initilization for each weight
        pass

    def char_emb_layer(self, x):
        """
        :param x: (batch, seq_len, word_len)
        :return: (batch, seq_len, char_channel_size)
        """
        batch_size = x.size(0)
        # (batch, seq_len, word_len, char_dim)
        x = self.char_emb(x)
        # (batch * seq_len, 1, char_dim, word_len)
        x = x.view(-1, self.args.char_dim, x.size(2)).unsqueeze(1)
        # (batch * seq_len, char_channel_size, 1, conv_len) -> (batch * seq_len, char_channel_size, conv_len)
        x = self.char_conv(x).squeeze()
        # (batch * seq_len, char_channel_size, 1) -> (batch * seq_len, char_channel_size)
        x = F.max_pool1d(x, x.size(2)).squeeze()
        # (batch, seq_len, char_channel_size)
        x = x.view(batch_size, -1, self.args.char_channel_size)

        return x

    def word_emb_layer(self, x):
        """
        :param x: (batch, seq_len)
        :return: (batch, seq_len, word_dim)
        """
        # (batch, seq_len, word_dim)
        return self.word_emb(x)

    def highway_layer(self, x1, x2):
        """
        :param x1: (batch, seq_len, char_channel_size)
        :param x2: (batch, seq_len, word_dim)
        :return: (batch, seq_len, hidden_size * 2)
        """
        # (batch, seq_len, char_channel_size + word_dim)
        x = torch.cat([x1, x2], dim=-1)
        for i in range(2):
            h = getattr(self, f'highway_linear{i}')(x)
            g = getattr(self, f'highway_gate{i}')(x)
            x = g * h + (1 - g) * x
        # (batch, seq_len, hidden_size * 2)
        return x

    def context_emb_layer(self, x):
        """
        :param x: (batch, seq_len, hidden_size * 2)
        :return: (batch, seq_len, hidden_size * 2)
        """
        # (batch, seq_len, hidden_size * 2)
        x, _ = self.context_LSTM(x)
        return x

    def att_flow_layer(self, c, q):
        """
        :param c: (batch, c_len, hidden_size * 2)
        :param q: (batch, q_len, hidden_size * 2)
        :return: (batch, c_len, q_len)
        """
        c_len = c.size(1)
        q_len = q.size(1)

        # (batch, c_len, q_len, hidden_size * 2)
        c_tiled = torch.stack([c] * q_len, dim=2)
        # (batch, c_len, q_len, hidden_size * 2)
        q_tiled = torch.stack([q] * c_len, dim=1)
        # (batch, c_len, q_len, hidden_size * 6)
        s = torch.cat([c_tiled, q_tiled, c_tiled * q_tiled], dim=-1)
        # (batch, c_len, q_len)
        s = self.att_weight(s).squeeze()

        # (batch, c_len, q_len)
        a = F.softmax(s, dim=2)
        # (batch, c_len, q_len) * (batch, q_len, hidden_size * 2) -> (batch, c_len, hidden_size * 2)
        c2q_att = torch.bmm(a, q)
        # (batch, c_len)
        b = F.softmax(torch.max(s, dim=2)[0], dim=1)
        # (batch, c_len) * (batch, c_len, hidden_size * 2) -> (batch, hidden_size * 2)
        q2c_att = torch.bmm(b.unsqueeze(1), c).squeeze()
        # (batch, c_len, hidden_size * 2) (tiled)
        q2c_att = torch.stack([q2c_att] * c_len, dim=1)

        # (batch, c_len, hidden_size * 8)
        x = torch.cat([c, c2q_att, c * c2q_att, c * q2c_att], dim=-1)
        return x

    def modeling_layer(self, x):
        """
        :param x: (batch, c_len, hidden_size * 8)
        :return: (batch, c_len, hidden_size * 2)
        """
        x, _ = self.modeling_LSTM(x)
        return x

    def output_layer(self, g, m):
        """
        :param g: (batch, c_len, hidden_size * 8)
        :param m: (batch, c_len ,hidden_size * 2)
        :return: p1: (batch, c_len), p2: (batch, c_len)
        """
        # (batch, c_len ,hidden_size * 10)
        p1 = torch.cat([g, m], dim=-1)
        # (batch, c_len)
        p1 = self.p1_weight(p1).squeeze()

        # (batch, c_len, hidden_size * 2)
        m2, _ = self.output_LSTM(m)
        # (batch, c_len, hidden_size * 10)
        p2 = torch.cat([g, m2], dim=-1)
        # (batch, c_len)
        p2 = self.p2_weight(p2).squeeze()

        return p1, p2

    def forward(self, batch):
        # 1. Character Embedding Layer
        c_char = self.char_emb_layer(batch.c_char)
        q_char = self.char_emb_layer(batch.q_char)
        # 2. Word Embedding Layer
        c_word = self.word_emb(batch.c_word)
        q_word = self.word_emb(batch.q_word)
        # Highway network
        c = self.highway_layer(c_char, c_word)
        q = self.highway_layer(q_char, q_word)
        # 3. Contextual Embedding Layer
        c = self.context_emb_layer(c)
        q = self.context_emb_layer(q)
        # 4. Attention Flow Layer
        g = self.att_flow_layer(c, q)
        # 5. Modeling Layer
        m = self.modeling_layer(g)
        # 6. Output Layer
        p1, p2 = self.output_layer(g, m)

        # (batch, c_len), (batch, c_len)
        return p1, p2
