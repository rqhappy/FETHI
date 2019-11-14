import torch.nn.functional as F
import torch.nn as nn
import torch
import util
import config


# class TAFET(nn.Module):
#     class LSTMEncoder(nn.Module):
#
#         def __init__(self, opt):
#             super(TAFET.LSTMEncoder, self).__init__()
#             self.in_dropout = nn.Dropout(p=.5)
#             self.lstm = nn.LSTM(config.EMBEDDING_DIM, config.LSTM_E_STATE_SIZE, batch_first=True)
#             self.out_dropout = nn.Dropout(p=.5)
#             self.batchnorm = nn.BatchNorm1d(config.LSTM_E_STATE_SIZE)
#
#         def forward(self, x):
#             # mention_c.shape = [batch, mention_c_len, embedding_dim]
#             xdrop = self.in_dropout(x)
#             _, (h_n, _) = self.lstm(xdrop)
#             outdrop = self.out_dropout(h_n)
#             outnorm = self.batchnorm(outdrop)
#             return outnorm.squeeze(0)  # [batch, rl]
#
#     class BiAttentionLSTMEncoder(nn.Module):
#
#         class AvgEncoder(nn.Module):
#
#             def __init__(self, opt):
#                 super(TAFET.BiAttentionLSTMEncoder.AvgEncoder, self).__init__()
#                 self.gate = nn.Sequential(nn.Linear(config.EMBEDDING_DIM, config.EMBEDDING_DIM), nn.Sigmoid())
#                 self.linear = nn.Sequential(nn.Linear(config.EMBEDDING_DIM, config.EMBEDDING_DIM), nn.Sigmoid())
#                 self.xbatchnorm = nn.BatchNorm1d(config.EMBEDDING_DIM)
#
#             def forward(self, mention, m_len):
#                 # mention.shape = [num, max_mention_len, embedding_dim]
#                 # m_len.shape = [num]
#                 x = self.xbatchnorm(mention.sum(-2).div(m_len.unsqueeze(-1)))
#                 highway_net = self.linear(x) * self.gate(x) + (1 - self.gate(x)) * x
#                 return torch.unsqueeze(highway_net, 1)  # [batch, 1, ra]
#
#         def __init__(self, opt):
#             super(TAFET.BiAttentionLSTMEncoder, self).__init__()
#
#             self.avg_encoder = TAFET.BiAttentionLSTMEncoder.AvgEncoder(opt)
#             self.in_dropout = nn.Dropout(p=.3)
#             # config.BERT_EMBEDDING_DIM
#             self.blstm = nn.LSTM(config.EMBEDDING_DIM, config.BALSTM_E_STATE_SIZE, num_layers=1,
#                                  batch_first=True, bidirectional=True)
#             self.out_dropout = nn.Dropout(p=.1)
#             self.attention = nn.Linear(config.BALSTM_E_STATE_SIZE, config.STRUCT_ATTEN_NUM,
#                                        bias=False)  # attention with structured attention
#             self.lstmBatchNorm = nn.BatchNorm1d(config.EMBEDDING_DIM)
#
#         def forward(self, lcon, mention, rcon):
#             # input of shape (batch, lcon+1+rcon, input_size)
#
#             data = [lcon, self.avg_encoder(mention[0], mention[1]), rcon]
#             if lcon is None:
#                 data = data[1:]
#             if rcon is None:
#                 data = data[:-1]
#
#             xdrop = self.lstmBatchNorm(self.in_dropout(torch.cat(data, 1)))
#
#             output, (_, _) = self.blstm(xdrop)
#             outdrop = self.out_dropout(output)
#             fh, bh = outdrop.split([config.BALSTM_E_STATE_SIZE, config.BALSTM_E_STATE_SIZE], -1)
#             H = (fh + bh)  # [batch, seq_len+1, hidden_size]
#             return H.transpose(-1, -2).matmul(F.softmax(self.attention(torch.tanh(H)), dim=1)) \
#                 .reshape([H.shape[0], -1, 1]).squeeze()
#
#     def __init__(self, opt):
#         super(TAFET, self).__init__()
#         device = torch.device(config.CUDA) if torch.cuda.is_available() and opt.cuda else 'cpu'
#         self.lstm_encoder, self.lbalstm_encoder, self.rbalstm_encoder = init_encoder(opt)
#
#         self.bceloss = nn.BCEWithLogitsLoss(reduction='sum')
#         self.num_of_cls = 113 if opt.corpus_dir == config.WIKI else 89
#
#         # self.type_atten_mat = torch.load(config.TYPE_ATTEN_FILE_PATH + opt.corpus_dir + "type_atten.pt"
#         #                                  , map_location=device).clone().detach().requires_grad_(True)
#
#         # self.type_atten_mat = nn.Parameter(self.type_atten_mat, requires_grad=True)
#         self.linear_encoder = nn.Linear(config.LINEAR_IN_SIZE, self.num_of_cls)
#         self.linear = nn.Linear(config.LINEAR_IN_SIZE, config.EMBEDDING_DIM)
#         nn.init.xavier_uniform_(self.linear.weight, gain=nn.init.calculate_gain('relu'))
#
#         self.type_atten = nn.Sequential(self.linear, nn.ReLU())
#         # self.S = torch.from_numpy(util.create_prior(
#         #     config.DATA_ROOT + opt.corpus_dir + config.TYPE_SET_INDEX_FILE)).float().to(device)
#         # self.v = torch.tensor([self.num_of_cls, config.LINEAR_IN_SIZE], dtype=torch.float, device=device)
#         # self.V = nn.Parameter(self.v, requires_grad=True)
#
#     def forward(self, input_data):
#
#         mention = [input_data[0], input_data[1]]
#         mention_neighbor = input_data[2]
#         lcontext, rcontext = input_data[3], input_data[4]
#
#         # representation = torch.cat((self.balstm_encoder(lcontext, mention, rcontext),
#         #                             self.lstm_encoder(mention_neighbor)), 1)
#
#         # representation = self.balstm_encoder(lcontext, mention, rcontext)
#         lrepr = self.lbalstm_encoder(lcontext, mention, None)
#         rrepr = self.rbalstm_encoder(None, mention, rcontext)
#         return self.linear_encoder(torch.cat([lrepr, rrepr], 1))
#
#         # return self.type_atten(representation).matmul(self.type_atten_mat.t())
#         # alpha = F.softmax(self.type_atten(representation).matmul(self.type_atten_mat.t()), dim=-1)
#         #
#         # #
#         # # print(f"alpha[0] = {alpha[0]}")
#         # return self.linear_encoder(self.type_linear(alpha))
#
#         # return self.type_atten(representation).matmul(self.type_atten_mat.t())
#         # return representation.matmul(self.S.matmul(self.V.t()))
#         # return self.linear_encoder(representation)
#
#     def get_bceloss(self):
#         return self.bceloss
#
#     def get_struct_loss(self):
#         a = self.lbalstm_encoder.attention.weight
#         b = self.rbalstm_encoder.attention.weight
#         aa = a.matmul(a.t())
#         bb = b.matmul(b.t())
#         i = torch.eye(aa.shape[0], dtype=torch.float, device=config.CUDA)
#         pa = torch.norm(aa - i, p='fro')
#         pb = torch.norm(bb - i, p='fro')
#         return pa * pa + pb * pb


class FET(nn.Module):



    # class LSTMEncoder(nn.Module):
    #
    #     def __init__(self, opt):
    #         super(FET.LSTMEncoder, self).__init__()
    #         self.in_dropout = nn.Dropout(p=config.DROPOUT)
    #         self.lstm = nn.LSTM(config.EMBEDDING_DIM, config.LSTM_E_STATE_SIZE, batch_first=True)
    #         self.out_dropout = nn.Dropout(p=config.DROPOUT)
    #         self.batchnorm = nn.BatchNorm1d((opt.batch_size, config.LSTM_E_STATE_SIZE))
    #
    #     def forward(self, x):
    #         # mention_c.shape = [batch, mention_c_len, embedding_dim]
    #         xdrop = self.in_dropout(x)
    #         _, (h_n, _) = self.lstm(xdrop)
    #         outdrop = self.out_dropout(h_n)
    #         # outnorm = self.batchnorm(outdrop)
    #         return outdrop.squeeze(0)  # [batch, rl]

    # class CharCNNEncoder(nn.Module):
    #     def __init__(self, opt):
    #         super(FET.CharCNNEncoder, self).__init__()
    #         self.char_cnn = nn.Conv2d(1, 1, config.KERNAL_SIZE)
    #         self.max_pool = nn.MaxPool2d(config.KERNAL_SIZE)
    #         self.batch_size = opt.batch_size
    #         self.in_dropout = nn.Dropout(p=config.DROPOUT)
    #         self.out_dropout = nn.Dropout(p=config.DROPOUT)
    #         self.char_embedding = nn.Sequential(nn.Linear(len(config.CHARS) + 2, config.CHAR_EMBEDDING_DIM, bias=False)
    #                                             , nn.ReLU())
    #         self.linear = nn.Sequential(
    #             nn.Linear(((config.CHAR_SEQ_PAD_LEN - config.KERNAL_SIZE + 1) // config.KERNAL_SIZE)
    #                       * ((config.CHAR_EMBEDDING_DIM - config.KERNAL_SIZE + 1) // config.KERNAL_SIZE)
    #                       , config.CHAR_OUT), nn.ReLU())
    #
    #     def forward(self, x):
    #         cnn_out = self.char_cnn(self.in_dropout(self.char_embedding(x).unsqueeze(dim=1)))
    #         return self.linear(self.max_pool(self.out_dropout(cnn_out)).view(x.shape[0], 1, -1)).squeeze()

    class CharLSTMEncoder(nn.Module):
        def __init__(self, opt):
            super(FET.CharLSTMEncoder, self).__init__()
            self.char_lstm = nn.LSTM(config.CHAR_EMBEDDING_DIM, config.CHARLSTM_E_STATE_SIZE, batch_first=True)
            self.char_embedding = nn.Linear(len(config.CHARS) + 2, config.CHAR_EMBEDDING_DIM, bias=False)
            self.char_linear = nn.Sequential(nn.Linear(config.CHARLSTM_E_STATE_SIZE, config.CHAR_OUT), nn.ReLU())
            self.in_dropout = nn.Dropout(p=config.DROPOUT)
            self.out_dropout = nn.Dropout(p=config.DROPOUT)

        def forward(self, mention_char):
            _, (output, _) = self.char_lstm(self.in_dropout(self.char_embedding(mention_char)))
            return self.out_dropout(self.char_linear(output.squeeze()))

    class BiAttentionLSTMEncoder(nn.Module):
        class AvgEncoder(nn.Module):

            def __init__(self, opt):
                super(FET.BiAttentionLSTMEncoder.AvgEncoder, self).__init__()

                # self.char_mention_encoder = FET.BiAttentionLSTMEncoder.AvgEncoder.CharLSTMEncoder(opt)
                self.gate = nn.Sequential(nn.Linear(config.EMBEDDING_DIM, config.EMBEDDING_DIM), nn.Sigmoid())
                self.linear = nn.Sequential(nn.Linear(config.EMBEDDING_DIM, config.EMBEDDING_DIM), nn.ReLU())

                self.map = nn.Sequential(nn.Linear(config.EMBEDDING_DIM*2, config.EMBEDDING_DIM), nn.ReLU())

            def forward(self, mention, m_len):
                # mention.shape = [num, max_mention_len, embedding_dim]
                # m_len.shape = [num]

                # char_mention_emb = self.char_mention_encoder(mention_char)
                x = mention.sum(-2).div(m_len.unsqueeze(-1))
                highway_net = self.linear(x) * self.gate(x) + x * (1 - self.gate(x))
                # highway_net = self.linear(char_mention_emb) * self.gate(char_mention_emb) \
                #               + (1 - self.gate(char_mention_emb)) * char_mention_emb

                # return torch.unsqueeze(self.map(torch.cat([x, highway_net], 1)), 1)  # [batch, 1, ra]
                return torch.unsqueeze(highway_net, 1)  # [batch, 1, ra]

        def __init__(self, opt):
            super(FET.BiAttentionLSTMEncoder, self).__init__()

            self.avg_encoder = FET.BiAttentionLSTMEncoder.AvgEncoder(opt)
            self.in_dropout = nn.Dropout(p=config.DROPOUT)
            self.blstm = nn.LSTM(config.EMBEDDING_DIM, config.BALSTM_E_STATE_SIZE, num_layers=1,
                                 batch_first=True, bidirectional=True)
            self.out_dropout = nn.Dropout(p=config.DROPOUT)

            self.attention = nn.Linear(config.BALSTM_E_STATE_SIZE, config.STRUCT_ATTEN_NUM,
                                       bias=False)  # structured attention

        def forward(self, lcon, mention, rcon):
            # input of shape (batch, lcon+1+rcon, input_size)

            data = [lcon, self.avg_encoder(mention[0], mention[1]), rcon]

            xdrop = self.in_dropout(torch.cat(data, 1))
            output, (_, _) = self.blstm(xdrop)
            outdrop = self.out_dropout(output)

            fh, bh = outdrop.split([config.BALSTM_E_STATE_SIZE, config.BALSTM_E_STATE_SIZE], -1)
            H = (fh + bh)  # [batch, seq_len+1, hidden_size]
            # 200*seq_len+1 x seq_len+1*2
            return H.transpose(-1, -2).matmul(F.softmax(self.attention(torch.tanh(H)), dim=1)) \
                .reshape([H.shape[0], -1, 1]).squeeze()

    def __init__(self, opt):
        super(FET, self).__init__()

        device = torch.device(config.CUDA) if torch.cuda.is_available() and opt.cuda else 'cpu'
        self.balstm_encoder, self.char_rnn_encoder = init_fet_encoder(opt)

        self.bcelogitsloss = nn.BCEWithLogitsLoss(reduction='sum')
        self.bceloss = nn.BCELoss(reduction="sum")
        self.num_of_cls = 113 if opt.corpus_dir == config.WIKI else 89

        self.infer_rnn = nn.RNN(config.INFER_DIM, config.INFER_DIM, nonlinearity="relu")
        self.dataset = opt.corpus_dir
        self.layers = [47, 66] if self.dataset == config.WIKI else [4, 44, 41]
        # self.layers = [43, 70]  # xu_hier
        self.w = torch.empty([len(self.layers), config.LINEAR_IN_SIZE, config.INFER_DIM], dtype=torch.float, device=device,
                             requires_grad=True)
        nn.init.xavier_uniform_(self.w, gain=nn.init.calculate_gain('relu'))
        self.W = nn.Parameter(self.w, requires_grad=True)

        self.linear_layer = nn.ModuleList([nn.Linear(config.INFER_DIM, x) for x in self.layers])

        # self.linear = nn.Linear(config.LINEAR_IN_SIZE, self.num_of_cls)
        # nn.init.xavier_uniform_(self.linear.weight, gain=nn.init.calculate_gain('relu'))

        self.in_drop = nn.Dropout(p=config.DROPOUT)
        self.out_drop = nn.Dropout(p=config.DROPOUT)

    def forward(self, input_data):
        mention = [input_data[0], input_data[1]]
        # mention_neighbor = input_data[2]
        lcontext, rcontext = input_data[2], input_data[3]
        mention_char = input_data[4]

        # balstm + char_lstm

        representation = torch.cat((self.balstm_encoder(lcontext, mention, rcontext),
                                    self.char_rnn_encoder(mention_char)), 1)

        # balstm + lstm
        # representation = torch.cat((self.balstm_encoder(lcontext, mention, rcontext, mention_char),
        #                             self.lstm_encoder(mention_neighbor)), 1)

        # balstmout = self.balstm_encoder(lcontext, mention, rcontext)
        # representation = torch.cat((self.balstm_encoder(lcontext, mention, rcontext),
        #                             self.lstm_encoder(mention_neighbor),
        #                             self.type_atten(balstmout).matmul(self.type_atten_mat.t())), 1)

        # balstm only
        # representation = self.balstm_encoder(lcontext, mention, rcontext)

        # balstm + feature
        # representation = torch.cat((self.balstm_encoder(lcontext, mention, rcontext),
        #                             self.feature_embeddings(feature)), 1)
        #

        # return self.first_linear(torch.cat((t[0].squeeze(), t[1].squeeze()), 1))

        # hier infer

        out, _ = self.infer_rnn(self.in_drop(representation.matmul(self.W).relu()))
        t = self.out_drop(out).split(1)

        return torch.cat([self.linear_layer[i](t[i].squeeze()) for i in range(len(self.layers))], 1)

        # return self.linear(representation)

    def get_bceloss(self):
        return self.bceloss

    def get_bcelogitsloss(self):
        return self.bcelogitsloss

    def get_layers(self):
        return self.layers

    def get_struct_loss(self):
        a = self.balstm_encoder.attention.weight
        aa = a.matmul(a.t())
        i = torch.eye(aa.shape[0], dtype=torch.float, device=config.CUDA)
        p = torch.norm(aa - i, p='fro')
        return p * p


def init_fet_encoder(opt):
    balstm_encoder = _init_balstm_encoder(opt)
    # lstm_encoder = _init_lstm_encoder(opt)
    # char_cnn_encoder = _init_char_cnn_encoder(opt)
    char_rnn_encoder = _init_char_rnn_encoder(opt)
    # return balstm_encoder, lstm_encoder, char_rnn_encoder, char_cnn_encoder
    # return balstm_encoder, char_rnn_encoder, char_cnn_encoder
    return balstm_encoder, char_rnn_encoder


def _init_balstm_encoder(opt):
    return FET.BiAttentionLSTMEncoder(opt)

def _init_char_rnn_encoder(opt):
    return FET.CharLSTMEncoder(opt)
