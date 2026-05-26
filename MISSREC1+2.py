import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from model_utils import Transformer
from sp_con_pag import RAFFusion  # 确保 sp_con_pag.py 中已包含 RAFFusion 类

class MISSRec(Transformer):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.train_stage = config['train_stage']
        self.temperature = config['temperature']
        self.lam = config['lambda']
        self.gamma = config['gamma']
        self.alpha = config['alpha']  # Alignment Loss 权重
        self.modal_type = config['modal_type']
        self.id_type = config['id_type']
        self.seq_mm_fusion = config['seq_mm_fusion']
        
        hidden_size = config['hidden_size']
        num_heads = config['num_heads']

        # 1. 模态适配器
        if 'text' in self.modal_type:
            self.text_adaptor = nn.Linear(config['plm_size'], hidden_size)
        if 'img' in self.modal_type:
            self.img_adaptor = nn.Linear(config['img_size'], hidden_size)

        # 2. Cross-Attention 交互模块 (用于文本和图像融合)
        if 'text' in self.modal_type and 'img' in self.modal_type:
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=num_heads,
                dropout=0.0,
                batch_first=True
            )
            # 3. 双路门控网络 (图文原始特征 + CA 交互特征)
            self.fusion_gate = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.Sigmoid()
            )

            # 4. Alignment Projection
            self.align_text_proj = nn.Linear(hidden_size, hidden_size // 2)
            self.align_img_proj = nn.Linear(hidden_size, hidden_size // 2)

        # 5. RAFFusion: 用于转导式微调阶段融合 ID 和 多模态特征
        if self.train_stage == 'transductive_ft' and self.id_type != 'none':
            # 这里的 num_heads 可以根据 hidden_size 调整，确保能整除
            self.raf_fusion = RAFFusion(dim=hidden_size, num_heads=6)

        # 基础组件初始化
        if self.train_stage in ['pretrain', 'inductive_ft']:
            self.item_embedding = None
        
        if self.train_stage in ['inductive_ft', 'transductive_ft']:
            all_num_embeddings = 0
            if 'text' in self.modal_type:
                self.plm_embedding = copy.deepcopy(dataset.plm_embedding)
                self.register_buffer('plm_embedding_empty_mask', (~self.plm_embedding.weight.data.sum(-1).bool()))
                all_num_embeddings += (self.plm_embedding.num_embeddings - 1)
                self.register_buffer('plm_interest_lookup_table', torch.zeros(self.plm_embedding.num_embeddings, dtype=torch.long))
            if 'img' in self.modal_type:
                self.img_embedding = copy.deepcopy(dataset.img_embedding)
                self.register_buffer('img_embedding_empty_mask', (~self.img_embedding.weight.data.sum(-1).bool()))
                all_num_embeddings += (self.img_embedding.num_embeddings - 1)
                self.register_buffer('img_interest_lookup_table', torch.zeros(self.img_embedding.num_embeddings, dtype=torch.long))

            self.num_interest = max(math.ceil(all_num_embeddings * config["interest_ratio"]), 1)
            self.register_buffer('interest_embeddings', torch.zeros(self.num_interest + 1, hidden_size, dtype=torch.float))

    def _cross_modal_fusion_with_align(self, text_emb, img_emb):
        """
        核心逻辑：Cross-Attention 交互 + 门控融合 + Alignment Loss
        """
        # A. Cross-Attention
        mm_interact, _ = self.cross_attn(query=text_emb, key=img_emb, value=img_emb)
        
        # B. 门控融合
        gate = self.fusion_gate(torch.cat([text_emb, mm_interact], dim=-1))
        fused_mm_emb = gate * text_emb + (1 - gate) * mm_interact

        # C. Alignment Loss (InfoNCE)
        t_feat = F.normalize(self.align_text_proj(text_emb.reshape(-1, text_emb.size(-1))), dim=-1)
        i_feat = F.normalize(self.align_img_proj(img_emb.reshape(-1, img_emb.size(-1))), dim=-1)
        
        logits = torch.matmul(t_feat, i_feat.transpose(0, 1)) / 0.07
        labels = torch.arange(t_feat.size(0), device=t_feat.device)
        align_loss = F.cross_entropy(logits, labels)
        
        return fused_mm_emb, align_loss

    def forward(self, item_seq, item_emb, item_modal_empty_mask, item_seq_len, interest_seq=None, interest_emb=None, interest_seq_len=None):
        # Decoder 输入准备
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_embedding = self.position_embedding(position_ids)
        
        # 1. 确定多模态特征表示 mm_representation
        if item_emb.dim() == 4:
            # 针对 'contextual' 模式，取第一个模态或平均，作为 RAFFusion 的输入
            mm_representation = item_emb.mean(1) 
        else:
            mm_representation = item_emb
        
        cur_epoch = getattr(self, 'cur_epoch', 0)
        # 2. ID Embedding 处理：采用 RAFFusion 策略
        if self.train_stage == 'transductive_ft' and self.id_type != 'none':
            item_id_emb = self.item_embedding(item_seq) # [B, L, D]

            if cur_epoch < 2: # 前 5 轮保护期：只做简单加法，不让 RAFFusion 干扰
                dec_input_emb = (item_id_emb + mm_representation) + position_embedding
            else:
            # 5 轮后 ID 基本对齐了，再开启复杂关系建模
                fused_offset = self.raf_fusion(item_id_emb, mm_representation)
                dec_input_emb = (item_id_emb + mm_representation) + 0.1 * fused_offset + position_embedding
            # 使用更合理的 RAFFusion 融合 ID 信息
            #fused_item_emb = self.raf_fusion(item_id_emb, mm_representation)
            #dec_input_emb = fused_item_emb + position_embedding
        else:
            dec_input_emb = mm_representation + position_embedding
            
        # 如果是多路上下文输入模式，保持维度展平逻辑
        if self.seq_mm_fusion != 'add':
            # 如果是 contextual，Transformer 需要处理展平后的序列
            dec_input_emb = dec_input_emb.view(dec_input_emb.size(0), -1, dec_input_emb.size(-1))
            
        dec_input_emb = self.LayerNorm(dec_input_emb)
        dec_input_emb = self.dropout(dec_input_emb)
        
        tgt_attn_mask, _, tgt_key_padding_mask = self.get_decoder_attention_mask(item_seq, item_modal_empty_mask, is_casual=False)
        src_attn_mask, src_key_padding_mask = self.get_encoder_attention_mask(interest_seq, is_casual=False)

        memory = self.trm_model.encoder(src=interest_emb, mask=src_attn_mask, src_key_padding_mask=src_key_padding_mask)
        
        # 兴趣正交正则项
        pooled_memory = (memory * (~src_key_padding_mask).unsqueeze(-1).float().mean(1, keepdim=True)).sum(1)
        interest_reg = (pooled_memory * pooled_memory).sum() / pooled_memory.shape[1]

        trm_output = self.trm_model.decoder(
            dec_input_emb, memory, tgt_mask=tgt_attn_mask, 
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask
        )

        output = self.gather_indexes(trm_output, item_seq_len - 1)
        return output, interest_reg.mean()

    def _compute_seq_embeddings(self, item_seq, item_seq_len):
        interest_seq_list = []
        text_emb, img_emb = None, None
        
        if 'text' in self.modal_type:
            text_emb = self.text_adaptor(self.plm_embedding(item_seq))
            text_mask = self.plm_embedding_empty_mask[item_seq]
            interest_seq_list.append(self.plm_interest_lookup_table[item_seq])
        if 'img' in self.modal_type:
            img_emb = self.img_adaptor(self.img_embedding(item_seq))
            img_mask = self.img_embedding_empty_mask[item_seq]
            interest_seq_list.append(self.img_interest_lookup_table[item_seq])

        if text_emb is not None and img_emb is not None:
            fused_mm, align_loss = self._cross_modal_fusion_with_align(text_emb, img_emb)
            item_emb_input = fused_mm if self.seq_mm_fusion == 'add' else torch.stack([text_emb, img_emb], dim=1)
            item_modal_empty_mask = torch.stack([text_mask, img_mask], dim=1)
        else:
            item_emb_input = text_emb if text_emb is not None else img_emb
            item_modal_empty_mask = (text_mask if text_emb is not None else img_mask).unsqueeze(1)
            align_loss = torch.tensor(0.0, device=item_seq.device)

        # 兴趣点提取逻辑
        all_interest_seq = torch.cat(interest_seq_list, dim=-1)
        unique_interest_seq = [s.unique() for s in all_interest_seq]
        unique_interest_len = torch.tensor([len(u) for u in unique_interest_seq], device=item_seq.device)
        unique_interest_seq = nn.utils.rnn.pad_sequence(unique_interest_seq, batch_first=True, padding_value=0)
        unique_interest_emb = self.interest_embeddings[unique_interest_seq]
        
        seq_output, reg = self.forward(
            item_seq, item_emb_input, item_modal_empty_mask, item_seq_len,
            unique_interest_seq, unique_interest_emb, unique_interest_len
        )
        return F.normalize(seq_output, dim=1), reg, align_loss

    def calculate_loss(self, interaction):
        if self.train_stage == 'pretrain':
            return super().calculate_loss(interaction)

        seq_output, interest_reg, align_loss = self._compute_seq_embeddings(interaction[self.ITEM_SEQ], interaction[self.ITEM_SEQ_LEN])
        
        test_item_emb = self._compute_test_item_embeddings()
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1)) / self.temperature
        
        rec_loss = self.loss_fct(logits, interaction[self.POS_ITEM_ID])
        # 最终 Loss 包含推荐 Loss + 兴趣正则 + 对齐 Loss
        return rec_loss + self.gamma * interest_reg + self.alpha * align_loss

    def _compute_test_item_embeddings(self):
        # 1. 模态基础融合
        if 'text' in self.modal_type and 'img' in self.modal_type:
            t_feat = self.text_adaptor(self.plm_embedding.weight).unsqueeze(1)
            i_feat = self.img_adaptor(self.img_embedding.weight).unsqueeze(1)
            fused_mm, _ = self._cross_modal_fusion_with_align(t_feat, i_feat)
            test_mm_emb = fused_mm.squeeze(1)
        else:
            test_mm_emb = self.text_adaptor(self.plm_embedding.weight) if 'text' in self.modal_type else self.img_adaptor(self.img_embedding.weight)

        # 2. 在测试阶段，如果是转导式，使用 RAFFusion 同步融合 ID
        if self.train_stage == 'transductive_ft' and self.id_type != 'none':
            test_id_emb = self.item_embedding.weight # [N, D]
            # 必须与训练时的融合逻辑一致
            test_item_emb = self.raf_fusion(test_id_emb, test_mm_emb)
        else:
            test_item_emb = test_mm_emb
            
        return F.normalize(test_item_emb, dim=1)

    def full_sort_predict(self, interaction):
        seq_output, _, _ = self._compute_seq_embeddings(interaction[self.ITEM_SEQ], interaction[self.ITEM_SEQ_LEN])
        test_item_emb = self._compute_test_item_embeddings()
        return torch.matmul(seq_output, test_item_emb.transpose(0, 1)) / self.temperature

    def get_encoder_attention_mask(self, dec_input_seq=None, is_casual=True):
        key_padding_mask = (dec_input_seq == 0)
        dec_seq_len = dec_input_seq.size(-1)
        attn_mask = torch.triu(torch.full((dec_seq_len, dec_seq_len), float('-inf'), device=dec_input_seq.device), diagonal=1) if is_casual else None
        return attn_mask, key_padding_mask

    def get_decoder_attention_mask(self, enc_input_seq, item_modal_empty_mask, is_casual=True):
        _, num_modality, seq_len = item_modal_empty_mask.shape
        if self.seq_mm_fusion == 'add':
            key_padding_mask = (enc_input_seq == 0)
        else:
            key_padding_mask = torch.logical_or((enc_input_seq == 0).unsqueeze(1), item_modal_empty_mask).flatten(1)
        attn_mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=enc_input_seq.device), diagonal=1) if is_casual else None
        if is_casual and self.seq_mm_fusion != 'add':
            attn_mask = torch.tile(attn_mask, (num_modality, num_modality))
        return attn_mask, None, key_padding_mask