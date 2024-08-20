import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math
from torch_geometric.nn import GATConv


# class CrossModalTemporalFusion(nn.Module):
#     def __init__(self, visual_input_dim, transformer_dim=128, nhead=8, num_layers=6):
#         super().__init__()
#         # 视觉特征提取：简化示例，实际使用时可能需要复杂的3D CNN或CNN+RNN结构
#         self.visual_encoder = nn.Sequential(
#             nn.Conv3d(visual_input_dim, transformer_dim, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3)),
#             nn.ReLU(),
#             nn.Flatten(start_dim=2),
#             nn.Linear(transformer_dim * 512, transformer_dim)  # 假设展平后的维度，根据实际调整
#         )
#
#         # 跨模态时序融合
#         self.fusion_transformer = nn.Transformer(d_model=transformer_dim, nhead=nhead, num_encoder_layers=num_layers)
#
#     def forward(self, visual_seqs, preprocessed_text_feats):
#         # 视觉序列处理：(batch_size, seq_len, C, H, W) -> (seq_len, batch_size, transformer_dim)
#         visual_features = self.visual_encoder(visual_seqs)
#         visual_features = visual_features.permute(1, 0, 2)  # 调整为(seq_len, batch_size, transformer_dim)
#
#         # 由于文本特征已经预处理且提取好，直接使用
#         # preprocessed_text_feats: (190, 8, 128)，已经是适当的维度，直接用于融合
#
#         # 融合并处理跨模态时序信息
#         combined_features = torch.cat([visual_features, preprocessed_text_feats], dim=0)  # 拼接视觉和文本特征
#         fusion_output = self.fusion_transformer(combined_features)
#
#         return fusion_output

class PositionalEncoding(nn.Module):
    def __init__(self, demb, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, demb, 2) * -(math.log(10000.0) / demb))
        pe = torch.zeros(max_len, 1, demb)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class SelfAttn(nn.Module):
    '''
    self-attention with learnable parameters
    '''

    def __init__(self, dhid):
        super().__init__()
        self.scorer = nn.Linear(dhid, 1)

    def forward(self, inp):
        scores = F.softmax(self.scorer(inp), dim=1)
        cont = scores.transpose(1, 2).bmm(inp).squeeze(1)
        return cont


class DotAttn(nn.Module):
    '''
    dot-attention (or soft-attention)
    '''

    def forward(self, inp, h):
        score = self.softmax(inp, h)
        return score.expand_as(inp).mul(inp).sum(1), score

    def softmax(self, inp, h):
        raw_score = inp.bmm(h.unsqueeze(2))
        score = F.softmax(raw_score, dim=1)
        return score


class ResnetVisualEncoder(nn.Module):
    '''
    visual encoder
    '''

    def __init__(self, dframe):
        super(ResnetVisualEncoder, self).__init__()
        self.dframe = dframe
        self.flattened_size = 64*7*7

        self.conv1 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0)
        self.fc = nn.Linear(self.flattened_size, self.dframe)
        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(64)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))

        x = self.conv2(x)
        x = F.relu(self.bn2(x))

        x = x.view(-1, self.flattened_size)
        x = self.fc(x)

        return x

class CrossModalAttention(nn.Module):
    '''
    Cross-modal attention module for combining visual and linguistic features
    '''
    def __init__(self, d_model, nhead, dim_feedforward):
        super().__init__()
        self.transformer_cross_modal = TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward
        )

    def forward(self, vis_feats, lang_feats):
        # Assuming vis_feats and lang_feats are of shape (S, N, E)
        combined_feats = torch.cat([vis_feats, lang_feats], dim=0)
        return self.transformer_cross_modal(combined_feats)

class ImprovedResnetVisualEncoder(nn.Module):
    """
    包括对序列视觉信息的改进处理。
    """
    def __init__(self, dframe, rnn_hidden_size=128):
        super().__init__()
        self.dframe = dframe
        self.rnn_hidden_size = rnn_hidden_size
        self.flattened_size = 64 * 7 * 7

        # CNN部分
        self.conv1 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0)
        self.fc = nn.Linear(self.flattened_size, self.dframe)
        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(64)

        # LSTM部分
        self.lstm = nn.LSTM(input_size=dframe, hidden_size=self.rnn_hidden_size, batch_first=True)
    def forward(self, x):
        batch_size, seq_len, C, H, W = x.size() # (8,117,512,7,7)

        # 保持x的原始批量和序列维度不变，处理每个帧
        x = x.view(batch_size * seq_len, C, H, W) # (936,512,7,7)

        conv1_out = self.conv1(x) # (936,256,7,7)
        x = F.relu(self.bn1(conv1_out)) # (936,256,7,7)
        conv2_out = self.conv2(x) # (936,64,7,7)
        x = F.relu(self.bn2(conv2_out)) # (936,64,7,7)
        x = x.view(-1, self.flattened_size) # (936, 3136)
        x = self.fc(x) # (936,128)

        # 将CNN的输出重新排列成(batch_size, seq_len, -1)，以匹配RNN的输入
        x = x.view(batch_size, seq_len, -1) # (8,117,128)
        # 使用RNN处理整个帧序列，假设batch_first=True
        output, _ = self.lstm(x) # (8,117,128)
        # 为了匹配期望的输出维度(117, 8, 128)，确保seq_len = 117, rnn_hidden_size = 128
        return output, conv1_out, conv2_out  # (8, 117, 128), (936,256,7,7), (936,64,7,7)


# class CNNTransformerVisualEncoder(nn.Module):
#     """
#     包括对序列视觉信息的改进处理的类，使用 Transformer 模型处理序列。
#     """
#     def __init__(self, dframe, rnn_hidden_size=128, num_heads=4, num_layers=2, dropout=0.1):
#         super().__init__()
#         self.dframe = dframe
#         self.rnn_hidden_size = rnn_hidden_size
#         self.flattened_size = 64 * 7 * 7
#         # CNN部分
#         self.conv1 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
#         self.conv2 = nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0)
#         self.fc = nn.Linear(self.flattened_size, self.dframe)
#         self.bn1 = nn.BatchNorm2d(256)
#         self.bn2 = nn.BatchNorm2d(64)
#         # Transformer部分
#         self.transformer = nn.Transformer(
#             d_model=self.dframe,
#             nhead=num_heads,
#             num_encoder_layers=num_layers,
#             num_decoder_layers=num_layers,
#             dim_feedforward=self.rnn_hidden_size * 4,
#             dropout=dropout,
#             batch_first=True
#         )
#     def forward(self, x):
#         batch_size, seq_len, C, H, W = x.size() # (8,117,512,7,7)
#         # 处理每个帧，保持原始批量和序列维度
#         x = x.view(batch_size * seq_len, C, H, W) # (936,512,7,7)
#         conv1_out = self.conv1(x) # (936,256,7,7)
#         x = F.relu(self.bn1(conv1_out)) # (936,256,7,7)
#         conv2_out = self.conv2(x) # (936,64,7,7)
#         x = F.relu(self.bn2(conv2_out)) # (936,64,7,7)
#         x = x.view(-1, self.flattened_size) # (936, 3136)
#         x = self.fc(x) # (936,128)
#         # 将CNN的输出重新排列成(batch_size, seq_len, -1)并加上位置编码
#         x = x.view(batch_size, seq_len, -1) # (8,117,128)
#         # 动态创建位置编码并加到x上
#         x += self.create_positional_encoding(seq_len, self.dframe, x.device)
#         # 使用 Transformer 处理整个帧序列
#         output = self.transformer(x, x) # (8,117,128)
#         return output, conv1_out, conv2_out # (8, 117, 128), (936,256,7,7), (936,64,7,7)
#     def create_positional_encoding(self, seq_len, demb, device):
#         position = torch.arange(seq_len, device=device).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, demb, 2, device=device) * -(math.log(10000.0) / demb))
#         pe = torch.zeros(seq_len, 1, demb, device=device)
#         pe[:, 0, 0::2] = torch.sin(position * div_term)
#         pe[:, 0, 1::2] = torch.cos(position * div_term)
#         return pe.squeeze(1)


class SpatialPyramidPooling(nn.Module):
    """ 空间金字塔池化模块 """
    def __init__(self, levels=[1, 2, 4]):
        super().__init__()
        self.levels = levels

    def forward(self, x):
        n, c, h, w = x.size()
        features = []
        for level in self.levels:
            kernel_size = (h // level, w // level)
            stride = kernel_size
            pooling = F.avg_pool2d(x, kernel_size, stride)
            features.append(F.interpolate(pooling, size=(h, w), mode='bilinear', align_corners=False))
        return torch.cat(features, dim=1)  # Concatenate along channel dimension

class SelfAttention(nn.Module):
    """ 简单的自注意力层 """
    def __init__(self, in_channels):
        super().__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.softmax = nn.Softmax(dim=-1)  # Softmax over width*height

    def forward(self, x):
        batch, channels, height, width = x.size()
        query = self.query_conv(x).view(batch, -1, height*width).permute(0, 2, 1)
        key = self.key_conv(x).view(batch, -1, height*width)
        value = self.value_conv(x).view(batch, -1, height*width)

        attention = torch.bmm(query, key)  # Batch matrix multiplication
        attention = self.softmax(attention)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch, channels, height, width)
        return out + x  # Add input for residual connection

class CNNTransformerVisualEncoder(nn.Module):
    def __init__(self, dframe, rnn_hidden_size=128, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.dframe = dframe
        self.rnn_hidden_size = rnn_hidden_size
        self.spp = SpatialPyramidPooling(levels=[1, 2, 4])
        # Expanded channel count to account for SPP output
        self.conv1 = nn.Conv2d(512 * 4, 256, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0)
        self.fc = nn.Linear(64 * 7 * 7, self.dframe)
        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(64)
        self.attention = SelfAttention(512)
        self.transformer = nn.Transformer(
            d_model=self.dframe,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=self.rnn_hidden_size * 4,
            dropout=dropout,
            batch_first=True
        )

    def forward(self, x):
        batch_size, seq_len, C, H, W = x.size()
        x = x.view(batch_size * seq_len, C, H, W)
        x = self.attention(x)
        x = self.spp(x)
        conv1_out = self.conv1(x)
        x = F.relu(self.bn1(conv1_out))
        conv2_out = self.conv2(x)
        x = F.relu(self.bn2(conv2_out))
        x = x.view(-1, 64 * 7 * 7)
        x = self.fc(x)
        x = x.view(batch_size, seq_len, -1)
        x += self.create_positional_encoding(seq_len, self.dframe, x.device)
        output = self.transformer(x, x)
        return output, conv1_out, conv2_out

    def create_positional_encoding(self, seq_len, demb, device):
        position = torch.arange(seq_len, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, demb, 2, device=device) * -(math.log(10000.0) / demb))
        pe = torch.zeros(seq_len, 1, demb, device=device)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        return pe.squeeze(1)

class MaskDecoder(nn.Module):
    '''
    mask decoder
    '''

    def __init__(self, dhid, pframe=300, hshape=(64,7,7)):
        super(MaskDecoder, self).__init__()
        self.dhid = dhid
        self.hshape = hshape
        self.pframe = pframe

        self.d1 = nn.Linear(self.dhid, hshape[0]*hshape[1]*hshape[2])
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn1 = nn.BatchNorm2d(16)
        self.dconv3 = nn.ConvTranspose2d(64 + 256, 32, kernel_size=4, stride=2, padding=1)
        self.dconv2 = nn.ConvTranspose2d(32 + 64, 16, kernel_size=4, stride=2, padding=1)
        self.dconv1 = nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1)

    def forward(self, x, conv1_feat, conv2_feat): # (8,117,128), (936,256,7,7), (936,64,7,7)
        batch_size, seq_len = x.shape[0], x.shape[1]
        x = F.relu(self.d1(x)) # (8,117,3136)
        x = x.view(-1, *self.hshape) # (936,64,7,7)

        # 从编码器获取的特征
        conv1_feat = F.interpolate(conv1_feat, size=(14, 14), mode='nearest') # (936,256,14,14)
        conv2_feat = F.interpolate(conv2_feat, size=(56, 56), mode='nearest') #  (936,64,56,56)

        x = self.upsample(x) # (936,64,14,14)
        x = torch.cat([x, conv1_feat], dim=1) # (936,320,14,14)
        x = self.dconv3(x) # (936,32,28,28)
        x = F.relu(self.bn2(x)) # (936,32,28,28)

        x = self.upsample(x) # (936,32,56,56)
        x = torch.cat([x, conv2_feat], dim=1) # (936,98,56,56)
        x = self.dconv2(x) # (936,16,112,112)
        x = F.relu(self.bn1(x)) # (936,16,112,112)

        x = self.dconv1(x) # (936,1,224,224)
        x = F.interpolate(x, size=(self.pframe, self.pframe), mode='bilinear')  # (936,1,300,300)

        x = x.view(batch_size, seq_len, 1, self.pframe, self.pframe) # (8,117,1,300,300)

        return x



class GATLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GATLayer, self).__init__()
        self.gat = GATConv(in_channels, out_channels, heads=1)  # 单头GAT

    def forward(self, x, edge_index):
        return self.gat(x, edge_index)

class GATFeatureExtractor(nn.Module):
    def __init__(self, feature_size):
        super(GATFeatureExtractor, self).__init__()
        self.gat1 = GATLayer(feature_size, feature_size)  # GAT层保持输入输出维度一致

    def forward(self, x, edge_index):
        # x的形状应为[N, F]，N为节点数，F为每个节点的特征数
        # edge_index的形状为[2, E]，E为边数
        return self.gat1(x, edge_index)

def create_linear_edge_index(num_nodes):
    # 创建连接相邻节点的边
    # num_nodes是序列长度，即117
    edges = []
    for i in range(num_nodes):
        if i > 0:
            edges.append([i, i-1])
        if i < num_nodes - 1:
            edges.append([i, i+1])
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


class ConvFrameMaskDecoderProgressMonitor(nn.Module):
    '''
    action decoder with subgoal and progress monitoring
    '''

    def __init__(self, emb, dframe, dhid, pframe=300,
                actor_dropout=0., input_dropout=0.,
                 teacher_forcing=False):
        super().__init__()
        demb = emb.weight.size(1)

        self.emb = emb
        self.pframe = pframe
        self.dhid = dhid

        self.vis_encoder = CNNTransformerVisualEncoder(dframe=dframe, rnn_hidden_size=128)

        self.cross_modal_attention = CrossModalAttention(d_model=128, nhead=8, dim_feedforward=dhid)

        d_model = 128

        self.pos_encoder = PositionalEncoding(d_model)
        decoder_layer = TransformerDecoderLayer(d_model=d_model, nhead=8, dim_feedforward=dhid)
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=6)

        self.input_dropout = nn.Dropout(input_dropout)
        self.actor_dropout = nn.Dropout(actor_dropout)
        self.go = nn.Parameter(torch.Tensor(demb))

        self.actor = nn.Linear(d_model, len(emb.weight))
        self.mask_dec = MaskDecoder(dhid=d_model, pframe=self.pframe)
        self.teacher_forcing = teacher_forcing

        self.subgoal = nn.Linear(d_model, 1)
        self.progress = nn.Linear(d_model, 1)
        # 添加GAT特征提取器
        self.gat_feature_extractor = GATFeatureExtractor(1)  # 特征维度为1，对应(subgoal或progress)

        nn.init.uniform_(self.go, -0.1, 0.1)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, enc, frames, gold=None, max_decode=150):
        device = enc.device

        batch_size, seq_len, C, H, W = frames.size() # (8,117,512,7,7)

        # 直接处理整个帧序列，不需要逐帧处理和拼接
        vis_feats, conv1_feat, conv2_feat = self.vis_encoder(frames)  # (8, 117, 128), (936,256,7,7), (936,64,7,7)

        if vis_feats.size(-1) != 128:
            fc = nn.Linear(vis_feats.size(-1), 128).to(device)
            vis_feats = fc(vis_feats.view(batch_size * seq_len, -1))
            vis_feats = vis_feats.view(seq_len, batch_size, -1)
        else:
            # 如果已经是期望的维度，直接使用
            vis_feats = vis_feats.transpose(0, 1)  # 调整维度以匹配模型其他部分的期望 (batch_size, seq_len, rnn_hidden_size) (117, 8, 128)

        # 准备Transformer的memory，结合语言特征和视觉特征
        memory_lang = enc.transpose(0, 1)  # Transformer期望memory的维度为(S, N, E) (190, 8, 128)

        memory = self.cross_modal_attention(vis_feats, memory_lang)# 跨模态自注意力机制融合视觉特征和语言特征 (307, 8, 128)

        if gold is not None:
            tgt_seq = gold # (8,117),gold来自于feat的'action_low'
        else:
            tgt_seq = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)
            tgt_seq[:, 0] = self.go.squeeze()  # 使用GO token初始化第一个输入

        tgt_emb = self.emb(tgt_seq) # (8,117,128)
        tgt_emb = self.pos_encoder(tgt_emb) # (8,117,128)
        tgt_emb = tgt_emb.transpose(0, 1) # (117,8,128)

        tgt_mask = self.generate_square_subsequent_mask(seq_len).to(device) # (117,117)
        outs = self.transformer_decoder(tgt=tgt_emb, memory=memory, tgt_mask=tgt_mask) # (117,8,128)

        action_outs = self.actor(self.actor_dropout(outs)) # (117,8,15)

        mask_outs = self.mask_dec(outs.transpose(0, 1), conv1_feat, conv2_feat)  # (936,1,300,300)

        # 为序列长度创建edge_index
        edge_index = create_linear_edge_index(seq_len).to(device)  # 假设每个batch具有相同的序列长度

        subgoal_outs = torch.sigmoid(self.subgoal(outs.transpose(0, 1))) # (8,117,1)
        progress_outs = torch.sigmoid(self.progress(outs.transpose(0, 1))) # (8,117,1)

        # 扁平化处理，以适应GAT输入
        subgoal_outs_flat = subgoal_outs.view(-1, 1).to(device)  # (936, 1)
        progress_outs_flat = progress_outs.view(-1, 1).to(device)  # (936, 1)

        # 运行GAT
        subgoal_outs_gat = self.gat_feature_extractor(subgoal_outs_flat, edge_index)
        subgoal_outs_gat = subgoal_outs_gat.view(batch_size,-1,1)
        progress_outs_gat = self.gat_feature_extractor(progress_outs_flat, edge_index)
        progress_outs_gat = progress_outs_gat.view(batch_size,-1,1)

        results = {
            'out_action_low': action_outs.transpose(0, 1), # (8,117,15)
            'out_action_low_mask': mask_outs,  # (8,117,1,300,300)
            'out_subgoal': subgoal_outs_gat, #(8,117,1)
            'out_progress': progress_outs_gat, #(8,117,1)
        }
        return results

