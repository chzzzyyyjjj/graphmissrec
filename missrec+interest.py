import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from model_utils import Transformer


class MISSRec(Transformer):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        # 把 pre-computed LightGCN 存为 buffer
        try:
            import numpy as np
            import os
            emb_path = '/mnt/data/zyj/MM23-MISSRec/dataset/downstream/Scientific_mm_full/best_item_embedding.npy'
            if os.path.exists(emb_path):
                lightgcn_emb = np.load(emb_path)
                if lightgcn_emb.shape[0] == self.n_items:
                    lightgcn_tensor = torch.from_numpy(lightgcn_emb).float()
                elif lightgcn_emb.shape[0] == self.n_items - 1:
                    lightgcn_tensor = torch.from_numpy(
                        np.concatenate([np.zeros((1, lightgcn_emb.shape[1])), lightgcn_emb], axis=0)
                    ).float()
                else:
                    lightgcn_tensor = torch.zeros(self.n_items, self.hidden_size)
                
                # 恢复为不可学习的 buffer（完全截断梯度），保持 Graph 结构的纯粹性
                if lightgcn_tensor.shape[1] != self.hidden_size:
                    self.lightgcn_proj = nn.Linear(lightgcn_tensor.shape[1], self.hidden_size)
                self.register_buffer('lightgcn_graph_emb', lightgcn_tensor)
                self.graph_scale = nn.Parameter(torch.tensor(0.0, dtype=torch.float))
                self.interest_graph_scale = nn.Parameter(torch.tensor(0.0, dtype=torch.float))
        except Exception as e:
            print("Failed to load LightGCN embeddings:", e)

        self.train_stage = config['train_stage']
        self.temperature = config['temperature']
        self.lam = config['lambda']
        self.gamma = config['gamma']
        self.modal_type = config['modal_type']
        self.id_type = config['id_type']
        self.seq_mm_fusion = config['seq_mm_fusion'] # 'add' | 'contextual'
        self.interest_gate_net = nn.Linear(self.hidden_size, self.hidden_size)
        self.user_gate_net = nn.Linear(self.hidden_size, self.hidden_size)

        # 控制 graph_interest
        self.interest_gate = nn.Parameter(torch.tensor(-2.0))   # 全局强度（sigmoid后≈0.5起步）

# 可选：用户级偏好单独控制（更稳）
        self.user_pref_gate = nn.Parameter(torch.tensor(-1.0))  # 初始更小（sigmoid≈0.27）

# 可选：细粒度 gating（向量级）
        self.interest_vector_gate = nn.Linear(self.hidden_size, self.hidden_size)
        assert self.seq_mm_fusion in ['add', 'contextual']
        self.item_mm_fusion = config['item_mm_fusion'] # 'static' | 'dynamic_shared' | 'dynamic_instance'
        # NOTE: `plm_embedding` in pre-train stage will be carried via dataloader
        assert self.item_mm_fusion in ['static', 'dynamic_shared', 'dynamic_instance']

        assert self.train_stage in [
            'pretrain', 'inductive_ft', 'transductive_ft'
        ], f'Unknown train stage: [{self.train_stage}]'

        if self.train_stage in ['pretrain', 'inductive_ft']:
            self.item_embedding = None
        # for `transductive_ft`, `item_embedding` is defined in SASRec base model
        if self.train_stage in ['inductive_ft', 'transductive_ft']:
            # NOTE: `plm_embedding` in pre-train stage will be carried via dataloader
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

            # NOTE: 只在下游微调时起效
            self.num_interest = max(math.ceil(all_num_embeddings * config["interest_ratio"]), 1)
            self.knn_local_size = max(math.ceil(all_num_embeddings * config["knn_local_ratio"]), 1)
            self.register_buffer('interest_embeddings', torch.zeros(self.num_interest + 1, config['hidden_size'], dtype=torch.float))

        if 'text' in self.modal_type and 'img' in self.modal_type:
            if self.item_mm_fusion == 'dynamic_shared':
                self.fusion_factor = nn.Parameter(data=torch.tensor(0, dtype=torch.float))
            elif self.item_mm_fusion == 'dynamic_instance':
                self.fusion_factor = nn.Parameter(data=torch.zeros(self.n_items, dtype=torch.float))

        if 'text' in self.modal_type:
            self.text_adaptor = nn.Linear(config['plm_size'], config['hidden_size'])

        if 'img' in self.modal_type:
            self.img_adaptor = nn.Linear(config['img_size'], config['hidden_size'])

    def get_encoder_attention_mask(self, dec_input_seq=None, is_casual=True):
        """memory_mask: [BxL], dec_input_seq: [BxNq]"""
        key_padding_mask = (dec_input_seq == 0) # binary, [BxNq], Nq=L
        dec_seq_len = dec_input_seq.size(-1)
        attn_mask = torch.triu(torch.full((dec_seq_len, dec_seq_len), float('-inf'), device=dec_input_seq.device), diagonal=1) if is_casual else None
        return attn_mask, key_padding_mask

    def get_decoder_attention_mask(self, enc_input_seq, item_modal_empty_mask, is_casual=True):
        # enc_input_seq: [BxL]
        # item_modal_empty_mask: [BxMxL]
        assert enc_input_seq.size(0) == item_modal_empty_mask.size(0)
        assert enc_input_seq.size(-1) == item_modal_empty_mask.size(-1)
        batch_size, num_modality, seq_len = item_modal_empty_mask.shape # M
        if self.seq_mm_fusion == 'add':
            key_padding_mask = (enc_input_seq == 0) # binary, [BxL]
        else:
            # binary, [Bx1xL] | [BxMxL] => [BxMxL]
            key_padding_mask = torch.logical_or((enc_input_seq == 0).unsqueeze(1), item_modal_empty_mask)
            key_padding_mask = key_padding_mask.flatten(1) # [BxMxL] => [Bx(M*L)]
        if is_casual:
            attn_mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=enc_input_seq.device), diagonal=1) # [LxL]
            if self.seq_mm_fusion != 'add':
                attn_mask = torch.tile(attn_mask, (num_modality, num_modality)) # [(M*L)x(M*L)]
        else:
            attn_mask = None
        cross_attn_mask = None # Full mask
        return attn_mask, cross_attn_mask, key_padding_mask

    # def forward(self, enc_item_seq, item_emb, item_modal_empty_mask, item_seq_len, dec_input_seq=None, dec_input_emb=None, dec_inp_seq_len=None):
    def forward(self, item_seq, item_emb, item_modal_empty_mask, item_seq_len, interest_seq=None, interest_emb=None, interest_seq_len=None):
        # ===== Graph-aware Interest Enhancement =====
        if hasattr(self, 'lightgcn_graph_emb') and getattr(self, 'lightgcn_graph_emb', None) is not None:
            # [B, L, D]
            graph_emb_seq = self.lightgcn_graph_emb[item_seq]
            if hasattr(self, 'lightgcn_proj'):
                graph_emb_seq = self.lightgcn_proj(graph_emb_seq)
            graph_emb_seq = F.normalize(graph_emb_seq, dim=-1)

            # 用户级图偏好（全局）
            #graph_user_pref = graph_emb_seq.mean(dim=1)   # [B, D]

            # 融合（残差 + 可控）
            # 当前写定系数 0.1, 也可以后续放入 __init__ 设为 nn.Parameter
            if interest_emb is not None:
                attn_score = torch.matmul(interest_emb, graph_emb_seq.transpose(1, 2))  # [B, N_interest, L]
                attn_weight = F.softmax(attn_score, dim=-1)
                graph_interest = torch.matmul(attn_weight, graph_emb_seq)  # [B, N_interest, D]
                #interest_emb = interest_emb + 0.1 * graph_interest + 0.1 * graph_user_pref.unsqueeze(1)
                # ===== normalize（非常关键，防止graph主导）=====
                graph_interest = F.normalize(graph_interest, dim=-1)
                # interest-aware user preference
                attn_score = torch.matmul(interest_emb, graph_emb_seq.transpose(1, 2))  # [B, N, L]
                attn_weight = F.softmax(attn_score, dim=-1)
                graph_user_pref = torch.matmul(attn_weight, graph_emb_seq)  # [B, N, D]

# 必须降维成 user-level
                graph_user_pref = graph_user_pref.mean(dim=1)  # [B, D]
                #graph_user_pref = F.normalize(graph_user_pref, dim=-1)

# ===== gate（标量控制）=====
                interest_gate = torch.sigmoid(self.interest_gate)      # ∈ (0,1)
                user_gate = torch.sigmoid(self.user_pref_gate)

# ===== residual fusion =====
                interest_emb = interest_emb + \
                            interest_gate * graph_interest + \
                            user_gate * graph_user_pref.unsqueeze(1)

        # encoder input
        enc_input_emb = interest_emb
        src_attn_mask, src_key_padding_mask = self.get_encoder_attention_mask(interest_seq, is_casual=False)

        # ===== Graph-aware Interest Attention Bias =====
        if hasattr(self, 'lightgcn_graph_emb') and getattr(self, 'lightgcn_graph_emb', None) is not None and interest_emb is not None:
            # interest 之间的 graph 关系
            # 用 interest_emb 自身特征作为先验，构造相似度
            interest_norm = F.normalize(interest_emb, dim=-1)
            interest_graph_bias = torch.matmul(interest_norm, interest_norm.transpose(1, 2)) * self.interest_graph_scale # [B, N, N]

            num_heads = self.trm_model.encoder.layers[0].self_attn.num_heads
            interest_graph_bias = interest_graph_bias.repeat_interleave(num_heads, dim=0)

            if src_attn_mask is None:
                src_attn_mask = interest_graph_bias
            else:
                src_attn_mask = src_attn_mask.unsqueeze(0) + interest_graph_bias

        # decoder input
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_embedding = self.position_embedding(position_ids) # [LxD]
        dec_input_emb = item_emb + position_embedding # [BxMxLxD] or [BxLxD]
        if self.train_stage == 'transductive_ft':
            if self.id_type != 'none':
                item_id_embeddings = self.item_embedding(item_seq)
                if self.seq_mm_fusion != 'add':
                    item_id_embeddings = item_id_embeddings.unsqueeze(1) # [Bx1xLxD]
                dec_input_emb = dec_input_emb + item_id_embeddings
        if self.seq_mm_fusion != 'add':
            dec_input_emb = dec_input_emb.view(dec_input_emb.size(0), -1, dec_input_emb.size(-1)) # [BxMxLxD] => [Bx(M*L)xD]
        dec_input_emb = self.LayerNorm(dec_input_emb)
        dec_input_emb = self.dropout(dec_input_emb)
        tgt_attn_mask, tgt_cross_attn_mask, tgt_key_padding_mask = self.get_decoder_attention_mask(item_seq, item_modal_empty_mask, is_casual=False)
        memory_key_padding_mask = src_key_padding_mask

        # --- Inject Graph Bias directly into the attention mechanism ---
        if hasattr(self, 'lightgcn_graph_emb') and getattr(self, 'lightgcn_graph_emb', None) is not None:
            # graph_emb: [B, L, D]
            graph_emb_seq = self.lightgcn_graph_emb[item_seq]
            # 保证 pad = 0 被去除后的相似度稳定归一化
            graph_emb_seq = F.normalize(graph_emb_seq, dim=-1)
            # graph_bias: [B, L, L] -> similarity based on LightGCN
            graph_bias = torch.bmm(graph_emb_seq, graph_emb_seq.transpose(1, 2)) * self.graph_scale

            # Expand if sequence modalities are expanded (contextual, etc)
            if self.seq_mm_fusion != 'add':
                num_modality = item_modal_empty_mask.shape[1]
                graph_bias = graph_bias.repeat_interleave(num_modality, dim=1).repeat_interleave(num_modality, dim=2)
            
            # Since tgt_attn_mask is expected as [L*M, L*M] or [B*num_heads, L*M, L*M]
            num_heads = self.trm_model.decoder.layers[0].self_attn.num_heads
            # graph_bias: repeat to [B * num_heads, tgt_len, tgt_len]
            graph_bias = graph_bias.repeat_interleave(num_heads, dim=0)

            if tgt_attn_mask is None:
                tgt_attn_mask = graph_bias
            else:
                tgt_attn_mask = tgt_attn_mask.unsqueeze(0) + graph_bias
        # ---------------------------------------------------------------

        # contextual encoder
        memory = self.trm_model.encoder(src=enc_input_emb, mask=src_attn_mask, src_key_padding_mask=src_key_padding_mask)

        # diversity regularization for interest tokens
        src_key_token_weight = (~src_key_padding_mask).unsqueeze(-1).float().mean(1, keepdim=True) # [BxL] => [BxLx1] => [Bx1x1]
        pooled_memory = (memory * src_key_token_weight).sum(1)  # ([BxLxD] * [Bx1x1]) => [BxD]
        interest_orthogonal_regularization = (pooled_memory * pooled_memory).sum() / pooled_memory.shape[1] # [BxD] x [BxD] => [B]

        # interest-aware decoder
        trm_output = self.trm_model.decoder(
            dec_input_emb, memory, tgt_mask=tgt_attn_mask, memory_mask=tgt_cross_attn_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )

        output = self.gather_indexes(trm_output, item_seq_len - 1)
        return output, interest_orthogonal_regularization.mean()  # [BxD], []

    def seq_item_contrastive_task(self, seq_output, interaction, batch_labels):
        if 'text' in self.modal_type:
            pos_text_emb = self.text_adaptor(interaction['pos_text_emb'])
        if 'img' in self.modal_type:
            pos_img_emb = self.img_adaptor(interaction['pos_img_emb'])
        if 'text' in self.modal_type and 'img' in self.modal_type: # weighted fusion
            logits = self._compute_dynamic_fused_logits(seq_output, pos_text_emb, pos_img_emb) / self.temperature
        else: # single modality or no modality
            if 'text' in self.modal_type:
                pos_item_emb = pos_text_emb
            if 'img' in self.modal_type:
                pos_item_emb = pos_img_emb
            pos_items_emb = F.normalize(pos_items_emb, dim=1)
            logits = torch.matmul(seq_output, pos_item_emb.transpose(0, 1)) / self.temperature
        loss = F.cross_entropy(logits, batch_labels)
        return loss

    def seq_seq_contrastive_task(self, seq_output, interaction, img_emb, batch_labels):
        seq_output_aug, interest_orthogonal_regularization_aug = self._compute_seq_embeddings_pretrain(
            item_seq=interaction[self.ITEM_SEQ + '_aug'], 
            item_seq_len=interaction[self.ITEM_SEQ_LEN + '_aug'], 
            text_emb=self.text_adaptor(interaction['text_emb_list_aug']),
            img_emb=img_emb, 
            text_emb_empty_mask=interaction['text_emb_empty_mask_list_aug'],
            img_emb_empty_mask=interaction['img_emb_empty_mask_list'],
            # text_interest_seq=interaction['text_interest_list_aug'],
            # img_interest_seq=interaction['img_interest_list'],
            unique_interest_seq=interaction['unique_interest_list_aug'], 
            unique_interest_emb_list=interaction['unique_interest_emb_list_aug'], 
            unique_interest_len=interaction['unique_interest_len_aug']
        )
        logits = torch.matmul(seq_output, seq_output_aug.transpose(0, 1)) / self.temperature
        loss = F.cross_entropy(logits, batch_labels)
        return loss, interest_orthogonal_regularization_aug

    def pretrain(self, interaction):
        img_emb=self.img_adaptor(interaction['img_emb_list'])
        seq_output, interest_orthogonal_regularization = self._compute_seq_embeddings_pretrain(
            item_seq=interaction[self.ITEM_SEQ], 
            item_seq_len=interaction[self.ITEM_SEQ_LEN], 
            text_emb=self.text_adaptor(interaction['text_emb_list']),
            img_emb=img_emb, 
            text_emb_empty_mask=interaction['text_emb_empty_mask_list'],
            img_emb_empty_mask=interaction['img_emb_empty_mask_list'],
            # text_interest_seq=interaction['text_interest_list'],
            # img_interest_seq=interaction['img_interest_list'],
            unique_interest_seq=interaction['unique_interest_list'], 
            unique_interest_emb_list=interaction['unique_interest_emb_list'], 
            unique_interest_len=interaction['unique_interest_len']
        )
        batch_size = seq_output.shape[0]
        device = seq_output.device
        batch_labels = torch.arange(batch_size, device=device, dtype=torch.long)
        
        loss_seq_item = self.seq_item_contrastive_task(seq_output, interaction, batch_labels)
        loss_seq_seq, interest_orthogonal_regularization_aug = self.seq_seq_contrastive_task(
            seq_output, interaction, img_emb, batch_labels)
        loss = loss_seq_item + self.lam * loss_seq_seq + self.gamma * (interest_orthogonal_regularization + interest_orthogonal_regularization_aug)
        return loss

    def _compute_seq_embeddings_pretrain(
            self, item_seq, item_seq_len, 
            text_emb, img_emb, 
            text_emb_empty_mask=None,
            img_emb_empty_mask=None,
            # text_interest_seq=None,
            # img_interest_seq=None
            unique_interest_seq=None, 
            unique_interest_emb_list=None, 
            unique_interest_len=None
        ):

        item_emb_list = 0 if self.seq_mm_fusion == 'add' else []
        item_modal_empty_mask_list = []
        # interest_seq_list = []
        if 'text' in self.modal_type:
            if self.seq_mm_fusion == 'add':
                item_emb_list = item_emb_list + text_emb
            else:
                item_emb_list.append(text_emb) # append [BxLxD]
            item_modal_empty_mask_list.append(text_emb_empty_mask) # append [BxL]
            # interest_seq_list.append(text_interest_seq)
        if 'img' in self.modal_type:
            if self.seq_mm_fusion == 'add':
                item_emb_list = item_emb_list + img_emb # [BxLxD]
            else:
                item_emb_list.append(img_emb) # append [BxLxD]
            item_modal_empty_mask_list.append(img_emb_empty_mask) # append [BxL]
            # interest_seq_list.append(img_interest_seq)
        if self.seq_mm_fusion != 'add':
            item_emb_list = torch.stack(item_emb_list, dim=1) # [BxMxLxD]
        item_modal_empty_mask = torch.stack(item_modal_empty_mask_list, dim=1) # [BxMxL]

        seq_output, interest_orthogonal_regularization = self.forward(
            item_seq=item_seq, 
            item_emb=item_emb_list, 
            item_modal_empty_mask=item_modal_empty_mask, 
            item_seq_len=item_seq_len, 
            interest_seq=unique_interest_seq, 
            interest_emb=unique_interest_emb_list, 
            interest_seq_len=unique_interest_len
        )
        seq_output = F.normalize(seq_output, dim=1)
        return seq_output, interest_orthogonal_regularization

    def _compute_seq_embeddings(self, item_seq, item_seq_len):
        if 'text' in self.modal_type:
            text_emb = self.text_adaptor(self.plm_embedding(item_seq))
            text_emb_empty_mask = self.plm_embedding_empty_mask[item_seq]
        if 'img' in self.modal_type:
            img_emb = self.img_adaptor(self.img_embedding(item_seq))
            img_emb_empty_mask = self.img_embedding_empty_mask[item_seq]

        # 改进4：把加法改成stack
        item_emb_list = 0 if self.seq_mm_fusion == 'add' else []
        item_modal_empty_mask_list = []
        interest_seq_list = []
        if 'text' in self.modal_type:
            if self.seq_mm_fusion == 'add':
                item_emb_list = item_emb_list + text_emb
            else:
                item_emb_list.append(text_emb) # append [BxLxD]
            item_modal_empty_mask_list.append(text_emb_empty_mask) # append [BxL]
            plm_interest_seq = self.plm_interest_lookup_table[item_seq] # [BxNq], Nq=L
            interest_seq_list.append(plm_interest_seq)
        if 'img' in self.modal_type:
            if self.seq_mm_fusion == 'add':
                item_emb_list = item_emb_list + img_emb # [BxLxD]
            else:
                item_emb_list.append(img_emb) # append [BxLxD]
            item_modal_empty_mask_list.append(img_emb_empty_mask) # append [BxL]
            img_interest_seq = self.img_interest_lookup_table[item_seq] # [BxNq], Nq=L
            interest_seq_list.append(img_interest_seq)
        if self.seq_mm_fusion != 'add':
            item_emb_list = torch.stack(item_emb_list, dim=1) # [BxMxLxD]
        item_modal_empty_mask = torch.stack(item_modal_empty_mask_list, dim=1) # [BxMxL]

        # deduplication
        unique_interest_seq = []
        unique_interest_len = []
        all_interest_seq = torch.cat(interest_seq_list, dim=-1)
        for sample in all_interest_seq:
            unique_interests = sample.unique()
            unique_interest_len.append(len(unique_interests))
            unique_interest_seq.append(unique_interests)
        unique_interest_seq = nn.utils.rnn.pad_sequence(unique_interest_seq, batch_first=True, padding_value=0)
        unique_interest_emb_list = self.interest_embeddings[unique_interest_seq] # [BxNqxD], Nq=L
        unique_interest_len = torch.tensor(unique_interest_len, device=unique_interest_seq.device)
        del interest_seq_list

        seq_output, interest_orthogonal_regularization = self.forward(
            item_seq=item_seq, 
            item_emb=item_emb_list, 
            item_modal_empty_mask=item_modal_empty_mask, 
            item_seq_len=item_seq_len, 
            interest_seq=unique_interest_seq, 
            interest_emb=unique_interest_emb_list, 
            interest_seq_len=unique_interest_len
        )
        seq_output = F.normalize(seq_output, dim=1)
        return seq_output, interest_orthogonal_regularization

    def _compute_test_item_embeddings(self):
        test_item_emb = 0
        if 'text' in self.modal_type:
            test_text_emb = self.text_adaptor(self.plm_embedding.weight)
            test_item_emb = test_item_emb + test_text_emb
        if 'img' in self.modal_type:
            test_img_emb = self.img_adaptor(self.img_embedding.weight)
            test_item_emb = test_item_emb + test_img_emb

        if self.train_stage == 'transductive_ft':
            if self.id_type != 'none':
                test_item_emb = test_item_emb + self.item_embedding.weight

        test_item_emb = F.normalize(test_item_emb, dim=1)
        return test_item_emb

    def _compute_dynamic_fused_logits(self, seq_output, text_emb, img_emb):
        text_emb = F.normalize(text_emb, dim=1)
        img_emb = F.normalize(img_emb, dim=1)
        text_logits = torch.matmul(seq_output, text_emb.transpose(0, 1)) # [BxB]
        img_logits = torch.matmul(seq_output, img_emb.transpose(0, 1)) # [BxB]
        modality_logits = torch.stack([text_logits, img_logits], dim=-1) # [BxBx2]
        if self.item_mm_fusion in ['dynamic_shared', 'dynamic_instance']:
            agg_logits = (modality_logits * F.softmax(modality_logits * self.fusion_factor.unsqueeze(-1), dim=-1)).sum(dim=-1) # [BxBx2] => [BxB]
        else: # 'static'
            agg_logits = modality_logits.mean(dim=-1) # [BxBx2] => [BxB]
        if self.train_stage == 'transductive_ft':
            if self.id_type != 'none':
                test_id_emb = F.normalize(self.item_embedding.weight, dim=1)
                id_logits = torch.matmul(seq_output, test_id_emb.transpose(0, 1))
                agg_logits = (id_logits + agg_logits * 2) / 3
        return agg_logits

    def calculate_loss(self, interaction):
        if self.train_stage == 'pretrain':
            return self.pretrain(interaction)

        # Loss for fine-tuning
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]

        seq_output, interest_orthogonal_regularization = self._compute_seq_embeddings(item_seq, item_seq_len)
        
        if 'text' in self.modal_type and 'img' in self.modal_type: # weighted fusion
            test_text_emb = self.text_adaptor(self.plm_embedding.weight)
            test_img_emb = self.img_adaptor(self.img_embedding.weight)
            logits = self._compute_dynamic_fused_logits(seq_output, test_text_emb, test_img_emb) / self.temperature
        else: # single modality or no modality
            test_item_emb = self._compute_test_item_embeddings()
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1)) / self.temperature

        pos_items = interaction[self.POS_ITEM_ID]
        loss = self.loss_fct(logits, pos_items) + self.gamma * interest_orthogonal_regularization
        return loss

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]

        seq_output, _ = self._compute_seq_embeddings(item_seq, item_seq_len)
        
        if 'text' in self.modal_type and 'img' in self.modal_type: # weighted fusion
            test_text_emb = self.text_adaptor(self.plm_embedding.weight)
            test_img_emb = self.img_adaptor(self.img_embedding.weight)
            scores = self._compute_dynamic_fused_logits(seq_output, test_text_emb, test_img_emb) / self.temperature
        else: # single modality or no modality
            test_item_emb = self._compute_test_item_embeddings()
            scores = torch.matmul(seq_output, test_item_emb.transpose(0, 1)) / self.temperature
        return scores