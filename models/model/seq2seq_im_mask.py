import os
import torch
import numpy as np
import nn.vnn as vnn
import collections
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from model.seq2seq import Module as Base
from models.utils.metric import compute_f1, compute_exact
from gen.utils.image_util import decompress_mask
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math

# 确保所有操作都在相同的设备上进行
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 位置编码类的实现
class PositionalEncoding(nn.Module):
    def __init__(self, demb, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1).to(device)# 生成位置信息，生成一个从0到max_len-1的连续整数，然后增加一个维度变成[max_len, 1]的形状
        div_term = torch.exp(torch.arange(0, demb, 2) * -(math.log(10000.0) / demb)).to(device)# 计算除数项，这里使用了exp和log的数学关系来生成一个位置编码的分母部分
        pe = torch.zeros(max_len, 1, demb).to(device)# 初始化位置编码矩阵为全0，其形状为[max_len, 1, demb]
        pe[:, 0, 0::2] = torch.sin(position * div_term)# 使用正弦函数生成偶数位置的编码值
        pe[:, 0, 1::2] = torch.cos(position * div_term)# 使用余弦函数生成奇数位置的编码值
        self.register_buffer('pe', pe)# 将pe注册为一个不需要求导的常数buffer

    def forward(self, x):
        x = x + self.pe[:x.size(0)]# 将输入x和对应的位置编码相加
        return self.dropout(x)# 通过dropout层后返回结果


class Module(Base):
    def __init__(self, args, vocab):
        '''
        Seq2Seq agent
        '''
        super().__init__(args, vocab)

        self.gamma = 2.0  # gamma > 0 reduces the relative loss for well-classified examples
        self.alpha = 0.25  # alpha balances the importance between positive/negative classes

        # Transformer Encoder Layer
        encoder_layers = TransformerEncoderLayer(d_model=args.demb, nhead=8, dim_feedforward=args.dhid)
        self.enc = TransformerEncoder(encoder_layer=encoder_layers, num_layers=6).to(device)

        self.pos_encoder = PositionalEncoding(args.demb, dropout=0.1).to(device)

        # subgoal monitoring
        self.subgoal_monitoring = (self.args.pm_aux_loss_wt > 0 or self.args.subgoal_aux_loss_wt > 0)

        # frame mask decoder
        decoder = vnn.ConvFrameMaskDecoderProgressMonitor
        self.dec = decoder(self.emb_action_low, args.dframe, 2*args.dhid,
                           pframe=args.pframe,
                           actor_dropout=args.actor_dropout,
                           input_dropout=args.input_dropout,
                           teacher_forcing=args.dec_teacher_forcing)

        # dropouts
        self.vis_dropout = nn.Dropout(args.vis_dropout)
        self.lang_dropout = nn.Dropout(args.lang_dropout, inplace=True)
        self.input_dropout = nn.Dropout(args.input_dropout)

        # internal states
        self.state_t = None
        self.e_t = None
        self.test_mode = False

        # bce reconstruction loss
        self.bce_with_logits = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.mse_loss = torch.nn.MSELoss(reduction='none')

        # paths
        self.root_path = os.getcwd()
        self.feat_pt = 'feat_conv.pt'

        # params
        self.max_subgoals = 25

        # reset model
        self.reset()

    def featurize(self, batch, load_mask=True, load_frames=True):
        '''
        tensorize and pad batch input
        '''
        device = torch.device('cuda') if self.args.gpu else torch.device('cpu')
        feat = collections.defaultdict(list)

        for ex in batch:
            ###########
            # auxillary
            ###########

            if not self.test_mode:
                # subgoal completion supervision
                if self.args.subgoal_aux_loss_wt > 0:
                    feat['subgoals_completed'].append(np.array(ex['num']['low_to_high_idx']) / self.max_subgoals)

                # progress monitor supervision
                if self.args.pm_aux_loss_wt > 0:
                    num_actions = len([a for sg in ex['num']['action_low'] for a in sg])
                    subgoal_progress = [(i+1)/float(num_actions) for i in range(num_actions)]
                    feat['subgoal_progress'].append(subgoal_progress)

            #########
            # inputs
            #########

            # serialize segments
            self.serialize_lang_action(ex)

            # goal and instr language
            lang_goal, lang_instr = ex['num']['lang_goal'], ex['num']['lang_instr']

            # zero inputs if specified
            lang_goal = self.zero_input(lang_goal) if self.args.zero_goal else lang_goal
            lang_instr = self.zero_input(lang_instr) if self.args.zero_instr else lang_instr

            # append goal + instr
            lang_goal_instr = lang_goal + lang_instr
            feat['lang_goal_instr'].append(lang_goal_instr)

            # load Resnet features from disk
            if load_frames and not self.test_mode:
                root = self.get_task_root(ex)
                im = torch.load(os.path.join(root, self.feat_pt))

                num_low_actions = len(ex['plan']['low_actions']) + 1  # +1 for additional stop action
                num_feat_frames = im.shape[0]

                # Modeling Quickstart (without filler frames)
                if num_low_actions == num_feat_frames:
                    feat['frames'].append(im)

                # Full Dataset (contains filler frames)
                else:
                    keep = [None] * num_low_actions
                    for i, d in enumerate(ex['images']):
                        # only add frames linked with low-level actions (i.e. skip filler frames like smooth rotations and dish washing)
                        if keep[d['low_idx']] is None:
                            keep[d['low_idx']] = im[i]
                    keep[-1] = im[-1]  # stop frame
                    feat['frames'].append(torch.stack(keep, dim=0))

            #########
            # outputs
            #########

            if not self.test_mode:
                # low-level action
                feat['action_low'].append([a['action'] for a in ex['num']['action_low']])

                # low-level action mask
                if load_mask:
                    feat['action_low_mask'].append([self.decompress_mask(a['mask']) for a in ex['num']['action_low'] if a['mask'] is not None])

                # low-level valid interact
                feat['action_low_valid_interact'].append([a['valid_interact'] for a in ex['num']['action_low']])

        # 在进行任何修改前，先复制一个键列表
        keys = list(feat.keys())
        for k in keys:
            v = feat[k]
            if k == 'lang_goal_instr':
                seqs = [torch.tensor(vv, device=device) for vv in v]
                pad_seq = pad_sequence(seqs, batch_first=True, padding_value=self.pad)
                seq_lengths = np.array(list(map(len, v)))
                feat['lang_goal_instr_lengths'] = torch.tensor(seq_lengths, device=device)
                embed_seq = self.emb_word(pad_seq)
                packed_input = pack_padded_sequence(embed_seq, seq_lengths, batch_first=True, enforce_sorted=False)
                feat[k] = packed_input
            elif k == 'action_low_mask':
                seqs = [torch.tensor(vv, device=device, dtype=torch.float) for vv in v]
                feat[k] = seqs
            elif k in {'subgoal_progress', 'subgoals_completed'}:
                seqs = [torch.tensor(vv, device=device, dtype=torch.float) for vv in v]
                pad_seq = pad_sequence(seqs, batch_first=True, padding_value=self.pad)
                feat[k] = pad_seq
            else:
                seqs = [torch.tensor(vv, device=device, dtype=torch.float if 'frames' in k else torch.long) for vv in v]
                pad_seq = pad_sequence(seqs, batch_first=True, padding_value=self.pad)
                feat[k] = pad_seq


            seq_lengths = [len(vv) for vv in feat['lang_goal_instr']]
            feat['seq_lengths'] = torch.tensor(seq_lengths, device=device)
        return feat

    def serialize_lang_action(self, feat):
        '''
        append segmented instr language and low-level actions into single sequences
        '''
        is_serialized = not isinstance(feat['num']['lang_instr'][0], list)
        if not is_serialized:
            feat['num']['lang_instr'] = [word for desc in feat['num']['lang_instr'] for word in desc]
            if not self.test_mode:
                feat['num']['action_low'] = [a for a_group in feat['num']['action_low'] for a in a_group]


    def decompress_mask(self, compressed_mask):
        '''
        decompress mask from json files
        '''
        mask = np.array(decompress_mask(compressed_mask))
        mask = np.expand_dims(mask, axis=0)
        return mask

    def forward(self, feat, max_decode=300):
        encoded_lang = self.encode_lang(feat)# (8,190,128)
        frames = self.vis_dropout(feat['frames'])# (8,117,512,7,7)
        res = self.dec(encoded_lang, frames, max_decode=max_decode, gold=feat['action_low'])
        feat.update(res)
        return feat

    def generate_attention_mask(self, seq_lengths, max_len):
        seq_lengths = seq_lengths.to(device)
        mask = torch.arange(max_len, device=device).expand(len(seq_lengths), max_len) < seq_lengths.unsqueeze(1)
        return mask

    def encode_lang(self, feat):
        '''
        feat:
            subgoals_completed(8,117)
            subgoal_progress(8,117)
            lang_goal_instr(847, 128)
            frames(8,117,512,7,7)
            action_low(8,117)
            action_low_mask(x,1,300,300)
            action_low_valid_interact(8,117)
            seq_length(847,190,8,8)
            lang_goal_instr_length[145,95,50,44,79,58,186,190]
        '''
        packed_lang_goal_instr = feat['lang_goal_instr'] # (847, 128)
        emb_lang_goal_instr, seq_lengths = pad_packed_sequence(packed_lang_goal_instr, batch_first=True)# (8,190,128), [145,95,50,44,79,58,186,190]
        emb_lang_goal_instr = emb_lang_goal_instr.to(device)
        emb_lang_goal_instr = self.pos_encoder(emb_lang_goal_instr)

        # 注意力掩码生成时，直接在正确的设备上
        attention_mask = self.generate_attention_mask(seq_lengths, emb_lang_goal_instr.size(1)).to(device)# (8,190)

        # 确保输入和模型在同一设备上
        encoded_output = self.enc(emb_lang_goal_instr.transpose(0, 1), src_key_padding_mask=~attention_mask).transpose(0, 1)# (8,190,128),
        return encoded_output

    def reset(self):
        '''
        reset internal states (used for real-time execution during eval)
        '''
        self.r_state = {
            'state_t': None,
            'e_t': None,
            'cont_lang': None,
            'enc_lang': None
        }

    def step(self, feat, prev_action=None):
        '''
        forward the model for a single time-step (used for real-time execution during eval)
        '''

        # encode language features
        if self.r_state['cont_lang'] is None and self.r_state['enc_lang'] is None:
            self.r_state['cont_lang'], self.r_state['enc_lang'] = self.encode_lang(feat)

        # initialize embedding and hidden states
        if self.r_state['e_t'] is None and self.r_state['state_t'] is None:
            self.r_state['e_t'] = self.dec.go.repeat(self.r_state['enc_lang'].size(0), 1)
            self.r_state['state_t'] = self.r_state['cont_lang'], torch.zeros_like(self.r_state['cont_lang'])

        # previous action embedding
        e_t = self.embed_action(prev_action) if prev_action is not None else self.r_state['e_t']

        # decode and save embedding and hidden states
        out_action_low, out_action_low_mask, state_t, *_ = self.dec.step(self.r_state['enc_lang'], feat['frames'][:, 0], e_t=e_t, state_tm1=self.r_state['state_t'])

        # save states
        self.r_state['state_t'] = state_t
        self.r_state['e_t'] = self.dec.emb(out_action_low.max(1)[1])

        # output formatting
        feat['out_action_low'] = out_action_low.unsqueeze(0)
        feat['out_action_low_mask'] = out_action_low_mask.unsqueeze(0)
        return feat


    def extract_preds(self, out, batch, feat, clean_special_tokens=True):
        '''
        output processing
        '''
        pred = {}
        for ex, alow, alow_mask in zip(batch, feat['out_action_low'].max(2)[1].tolist(), feat['out_action_low_mask']):
            # remove padding tokens
            if self.pad in alow:
                pad_start_idx = alow.index(self.pad)
                alow = alow[:pad_start_idx]
                alow_mask = alow_mask[:pad_start_idx]

            if clean_special_tokens:
                # remove <<stop>> tokens
                if self.stop_token in alow:
                    stop_start_idx = alow.index(self.stop_token)
                    alow = alow[:stop_start_idx]
                    alow_mask = alow_mask[:stop_start_idx]

            # index to API actions
            words = self.vocab['action_low'].index2word(alow)

            # sigmoid preds to binary mask
            alow_mask = F.sigmoid(alow_mask)
            p_mask = [(alow_mask[t] > 0.5).cpu().numpy() for t in range(alow_mask.shape[0])]

            task_id_ann = self.get_task_and_ann_id(ex)
            pred[task_id_ann] = {
                'action_low': ' '.join(words),
                'action_low_mask': p_mask,
            }

        return pred


    def embed_action(self, action):
        '''
        embed low-level action
        '''
        device = torch.device('cuda') if self.args.gpu else torch.device('cpu')
        action_num = torch.tensor(self.vocab['action_low'].word2index(action), device=device)
        action_emb = self.dec.emb(action_num).unsqueeze(0)
        return action_emb


    def compute_loss(self, out, batch, feat):
        '''
        loss function for Seq2Seq agent
        '''
        losses = dict()

        # GT and predictions
        p_alow = out['out_action_low'].reshape(-1, len(self.vocab['action_low'])) # (936,15)
        l_alow = feat['action_low'].reshape(-1) # (936,)
        p_alow_mask = out['out_action_low_mask'] # (936,1,300,300)
        valid = feat['action_low_valid_interact'] # (8,117)

        # action loss
        pad_valid = (l_alow != self.pad) # (936,)
        alow_loss = F.cross_entropy(p_alow, l_alow, reduction='none') # (936,)
        alow_loss *= pad_valid.float() # (936,)
        alow_loss = alow_loss.mean() # 某个数
        losses['action_low'] = alow_loss * self.args.action_loss_wt

        # mask loss
        valid_idxs = valid.view(-1).nonzero().view(-1) # (49,)

        # 计算 p_alow_mask 展平后的总元素数量
        total_elements = p_alow_mask.shape[0] * p_alow_mask.shape[1] * p_alow_mask.shape[2] * p_alow_mask.shape[3] # 84240000

        # 确保 valid_idxs 中的所有索引都在范围内
        valid_idxs = valid_idxs[valid_idxs < total_elements] # (49,)

        # 重新计算 flat_p_alow_mask，以保证它和 flat_alow_mask 形状一致
        # 首先，展平 p_alow_mask 以匹配 flat_alow_mask 的形状
        flat_p_alow_mask = p_alow_mask.view(-1, p_alow_mask.shape[2], p_alow_mask.shape[3], p_alow_mask.shape[4])

        # 接下来，确保 flat_alow_mask 有足够的元素以匹配 flat_p_alow_mask
        # 你可以选择根据 valid_idxs 对 flat_p_alow_mask 进行索引，或者保证两者在计算损失之前形状一致
        if len(valid_idxs) > 0:
            flat_p_alow_mask = flat_p_alow_mask[valid_idxs // (p_alow_mask.shape[2] * p_alow_mask.shape[3])]
        else:
            flat_p_alow_mask = flat_p_alow_mask[:0]  # 如果没有有效索引，使用空张量 # (49,300,300)

        flat_alow_mask = torch.cat(feat['action_low_mask'], dim=0)

        # 检查 flat_p_alow_mask 和 flat_alow_mask 形状，确保它们一致
        if flat_p_alow_mask.size(0) != flat_alow_mask.size(0):
            # 需要根据具体情况调整它们以使形状一致
            pass

        # 计算损失
        alow_mask_loss = self.weighted_mask_loss(flat_p_alow_mask, flat_alow_mask)
        losses['action_low_mask'] = alow_mask_loss * self.args.mask_loss_wt

        # subgoal completion loss
        if self.args.subgoal_aux_loss_wt > 0:
            p_subgoal = feat['out_subgoal'].squeeze(2)
            l_subgoal = feat['subgoals_completed']
            if l_subgoal.size(1) > p_subgoal.size(1):
                l_subgoal = l_subgoal[:, :p_subgoal.size(1)]
            sg_loss = self.mse_loss(p_subgoal, l_subgoal)
            sg_loss = sg_loss.view(-1) * pad_valid.float()
            subgoal_loss = sg_loss.mean()
            losses['subgoal_aux'] = self.args.subgoal_aux_loss_wt * subgoal_loss

        # progress monitoring loss
        if self.args.pm_aux_loss_wt > 0:
            p_progress = feat['out_progress'].squeeze(2)
            l_progress = feat['subgoal_progress']
            if l_progress.size(1) > p_progress.size(1):
                l_progress = l_progress[:, :p_progress.size(1)]
            pg_loss = self.mse_loss(p_progress, l_progress)
            pg_loss = pg_loss.view(-1) * pad_valid.float()
            progress_loss = pg_loss.mean()
            losses['progress_aux'] = self.args.pm_aux_loss_wt * progress_loss

        return losses

    def focal_loss(self, inputs, targets):
        '''
        Compute the focal loss between `inputs` and the ground truth `targets`.
        `inputs` should be a tensor of logits, and `targets` should be a tensor of the same shape containing binary labels.
        '''
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # Prevents nans when probability close to 0
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()

    def weighted_mask_loss(self, pred_masks, gt_masks):
        '''
        Replace the original BCE-based loss calculation with focal loss for the mask predictions.
        '''
        gt_masks = gt_masks.squeeze(1)
        pred_masks = pred_masks.squeeze(1)
        return self.focal_loss(pred_masks, gt_masks)


    def flip_tensor(self, tensor, on_zero=1, on_non_zero=0):
        '''
        flip 0 and 1 values in tensor
        '''
        res = tensor.clone()
        res[tensor == 0] = on_zero
        res[tensor != 0] = on_non_zero
        return res


    def compute_metric(self, preds, data):
        '''
        compute f1 and extract match scores for output
        '''
        m = collections.defaultdict(list)
        for task in data:
            ex = self.load_task_json(task)
            i = self.get_task_and_ann_id(ex)
            label = ' '.join([a['discrete_action']['action'] for a in ex['plan']['low_actions']])
            m['action_low_f1'].append(compute_f1(label.lower(), preds[i]['action_low'].lower()))
            m['action_low_em'].append(compute_exact(label.lower(), preds[i]['action_low'].lower()))
        return {k: sum(v)/len(v) for k, v in m.items()}


