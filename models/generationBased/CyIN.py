"""
From: https://github.com/RH-Lin/CyIN
Paper: CyIN: Cyclic Informative Latent Space for Bridging Complete and Incomplete Multimodal Learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import functools
from random import sample
import copy

from ..subNets.transformers_encoder.transformer import TransformerEncoder
from ..subNets import AlignSubNet

from transformers import AutoModel, AutoTokenizer
from transformers import BertModel, BertTokenizer, RobertaModel, RobertaTokenizer, XLNetModel, XLNetTokenizer, T5Model, T5Tokenizer, DebertaV2Tokenizer, DebertaV2Model
import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
TRANSFORMERS_MAP = {
    'bert': (BertModel, BertTokenizer),
    'roberta': (RobertaModel, RobertaTokenizer),
    'xlnet': (XLNetModel, XLNetTokenizer),
    't5': (T5Model, T5Tokenizer),
    'sbert': (AutoModel, AutoTokenizer),
    'deberta': (DebertaV2Model, DebertaV2Tokenizer)
}

__all__ = ['CyIN']

class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, -pred)
        n = torch.numel(diffs.data)
        mse = torch.sum(diffs.pow(2)) / n

        return mse

class Encoder(nn.Module):
    def __init__(self, input_dim, IB_inter_dim, output_dim):
        super(Encoder, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, IB_inter_dim, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv1d(IB_inter_dim, output_dim, kernel_size=1, padding=0),
        )

    def forward(self, x):
        return self.net(x)

class MLP(nn.Module):
    def __init__(self, input_dim, IB_inter_dim, output_dim):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, IB_inter_dim, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv1d(IB_inter_dim, output_dim, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

class ResidualAE(nn.Module):
    ''' Residual autoencoder using fc layers
        layers should be something like [128, 64, 32]
        eg:[128,64,32]-> add: [(input_dim, 128), (128, 64), (64, 32), (32, 64), (64, 128), (128, input_dim)]
                          concat: [(input_dim, 128), (128, 64), (64, 32), (32, 64), (128, 128), (256, input_dim)]
    '''
    def __init__(self, layers, n_blocks, input_dim, AE_kernel_size, dropout=0.5, use_bn=False):
        super(ResidualAE, self).__init__()
        self.use_bn = use_bn
        self.dropout = dropout
        self.n_blocks = n_blocks
        self.input_dim = input_dim
        self.AE_kernel_size = AE_kernel_size
        self.transition = nn.Sequential(
            nn.Conv1d(input_dim, input_dim, kernel_size=self.AE_kernel_size, stride=1, padding=self.AE_kernel_size // 2, bias=False),
            nn.ReLU(),
            nn.Conv1d(input_dim, input_dim, kernel_size=self.AE_kernel_size, stride=1, padding=self.AE_kernel_size // 2, bias=False)
        )
        for i in range(n_blocks):
            setattr(self, 'encoder_' + str(i), self.get_encoder(layers))
            setattr(self, 'decoder_' + str(i), self.get_decoder(layers))
    
    def get_encoder(self, layers):
        all_layers = []
        input_dim = self.input_dim
        for i in range(0, len(layers)):
            # all_layers.append(nn.Linear(input_dim, layers[i]))
            all_layers.append(nn.Conv1d(input_dim, layers[i], kernel_size=self.AE_kernel_size, stride=1, padding=self.AE_kernel_size // 2, bias=False))
            all_layers.append(nn.LeakyReLU())
            if self.use_bn:
                all_layers.append(nn.BatchNorm1d(layers[i]))
            if self.dropout > 0:
                all_layers.append(nn.Dropout(self.dropout))
            input_dim = layers[i]
        # delete the activation layer of the last layer
        decline_num = 1 + int(self.use_bn) + int(self.dropout > 0)
        all_layers = all_layers[:-decline_num]
        return nn.Sequential(*all_layers)
    
    def get_decoder(self, layers):
        all_layers = []
        decoder_layer = copy.deepcopy(layers)
        decoder_layer.reverse()
        decoder_layer.append(self.input_dim)
        for i in range(0, len(decoder_layer)-2):
            # all_layers.append(nn.Linear(decoder_layer[i], decoder_layer[i+1]))
            all_layers.append(nn.Conv1d(decoder_layer[i], decoder_layer[i+1], kernel_size=self.AE_kernel_size, stride=1, padding=self.AE_kernel_size // 2, bias=False))
            all_layers.append(nn.ReLU())
            if self.use_bn:
                all_layers.append(nn.BatchNorm1d(decoder_layer[i]))
            if self.dropout > 0:
                all_layers.append(nn.Dropout(self.dropout))
        
        # all_layers.append(nn.Linear(decoder_layer[-2], decoder_layer[-1])) 
        all_layers.append(nn.Conv1d(decoder_layer[-2], decoder_layer[-1], kernel_size=1, stride=1, padding=0, bias=False))
        return nn.Sequential(*all_layers)

    def forward(self, x):
        x_in = x
        x_out = x.clone().fill_(0)
        latents = []
        for i in range(self.n_blocks):
            encoder = getattr(self, 'encoder_' + str(i))
            decoder = getattr(self, 'decoder_' + str(i))
            x_in = x_in + x_out
            latent = encoder(x_in)
            x_out = decoder(latent)
            latents.append(latent)
        latents = torch.cat(latents, dim=-1)
        return self.transition(x_in+x_out), latents

class FcClassifier(nn.Module):
    def __init__(self, input_dim, layers, output_dim, dropout=0.3, use_bn=False):
        ''' Fully Connect classifier
            Parameters:
            --------------------------
            input_dim: input feature dim
            layers: [x1, x2, x3] will create 3 layers with x1, x2, x3 hidden nodes respectively.
            output_dim: output feature dim
            activation: activation function
            dropout: dropout rate
        '''
        super().__init__()
        self.all_layers = []
        for i in range(0, len(layers)):
            self.all_layers.append(nn.Linear(input_dim, layers[i]))
            self.all_layers.append(nn.ReLU())
            if use_bn:
                self.all_layers.append(nn.BatchNorm1d(layers[i]))
            if dropout > 0:
                self.all_layers.append(nn.Dropout(dropout))
            input_dim = layers[i]
        
        if len(layers) == 0:
            layers.append(input_dim)
            self.all_layers.append(Identity())
        
        self.fc_out = nn.Linear(layers[-1], output_dim)
        self.module = nn.Sequential(*self.all_layers)
    
    def forward(self, x):
        feat = self.module(x)
        out = self.fc_out(feat)
        return out, feat
    
class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
    
class CyIN(nn.Module):
    def __init__(self, args):
        super(CyIN, self).__init__()
        self.args = args
        self.device = args.device

        self.text_dim, self.audio_dim, self.vision_dim = args.feature_dims
        self.layers = args.nlevels
        self.num_heads = args.num_heads
        self.attn_dropout_l = args.attn_dropout_l
        self.attn_dropout_a = args.attn_dropout_a
        self.attn_dropout_v = args.attn_dropout_v
        self.attn_dropout_mem = args.attn_dropout_mem
        self.embed_dropout = args.embed_dropout
        self.attn_mask = args.attn_mask
        self.output_dropout = args.output_dropout

        self.common_dim = args.common_dim
        self.IB_inter_dim = args.IB_inter_dim
        self.IB_btnk_dim = args.IB_btnk_dim
        self.p_eta = args.p_eta

        self.RAE_layers = self.args.RAE_layers
        self.RAE_n_blocks = self.args.RAE_n_blocks
        self.RAE_kernel_size = 3
        self.use_cycle_reverse = True

        tokenizer_class = TRANSFORMERS_MAP[args.transformers][1]
        model_class = TRANSFORMERS_MAP[args.transformers][0]
        self.text_tokenizer = tokenizer_class.from_pretrained(
            pretrained_model_name_or_path='AffectiveComputing/pretrains/' + args.pretrained,
            do_lower_case=True)
        self.text_model = model_class.from_pretrained(
            pretrained_model_name_or_path='AffectiveComputing/pretrains/' + args.pretrained)

        self.alignNet = AlignSubNet(args, mode='avg_pool') # mode in ['avg_pool', 'ctc', 'conv1d']

        # token-level IB
        self.encoder_l = Encoder(input_dim=self.text_dim, IB_inter_dim=self.IB_inter_dim, output_dim=self.IB_btnk_dim * 2)
        self.encoder_a = Encoder(input_dim=self.audio_dim, IB_inter_dim=self.IB_inter_dim, output_dim=self.IB_btnk_dim * 2)
        self.encoder_v = Encoder(input_dim=self.vision_dim, IB_inter_dim=self.IB_inter_dim, output_dim=self.IB_btnk_dim * 2) 

        self.mlp_l = MLP(input_dim=self.IB_btnk_dim, IB_inter_dim=self.IB_inter_dim, output_dim=self.text_dim)
        self.mlp_a = MLP(input_dim=self.IB_btnk_dim, IB_inter_dim=self.IB_inter_dim, output_dim=self.audio_dim)
        self.mlp_v = MLP(input_dim=self.IB_btnk_dim, IB_inter_dim=self.IB_inter_dim, output_dim=self.vision_dim) 

        self.MSE_loss = nn.MSELoss()

        # MulT Fusion
        # 1. Temporal convolutional layers
        self.proj_l = nn.Conv1d(self.IB_btnk_dim, self.common_dim, kernel_size=args.conv1d_kernel_size, padding=args.padding, bias=False)
        self.proj_a = nn.Conv1d(self.IB_btnk_dim, self.common_dim, kernel_size=args.conv1d_kernel_size, padding=args.padding, bias=False)
        self.proj_v = nn.Conv1d(self.IB_btnk_dim, self.common_dim, kernel_size=args.conv1d_kernel_size, padding=args.padding, bias=False)
        

        # 2. Crossmodal Attentions
        self.trans_l_with_a = self.get_network(self_type='la')
        self.trans_l_with_v = self.get_network(self_type='lv')

        self.trans_a_with_l = self.get_network(self_type='al')
        self.trans_a_with_v = self.get_network(self_type='av')

        self.trans_v_with_l = self.get_network(self_type='vl')
        self.trans_v_with_a = self.get_network(self_type='va')

        # 3. Self Attentions
        self.trans_l_mem = self.get_network(self_type='l_mem', layers=3)
        self.trans_a_mem = self.get_network(self_type='a_mem', layers=3)
        self.trans_v_mem = self.get_network(self_type='v_mem', layers=3)

        combined_dim = 2 * 3 * self.common_dim
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        output_dim = args.num_classes if args.train_mode in ("recognition") else 1
        self.out_layer = nn.Linear(combined_dim, output_dim)
        self.out_layer_l = nn.Linear(self.IB_btnk_dim, output_dim)
        self.out_layer_a = nn.Linear(self.IB_btnk_dim, output_dim)
        self.out_layer_v = nn.Linear(self.IB_btnk_dim, output_dim)

        # Cyclic Translation RAE module
        self.netRAE_LA = ResidualAE(self.RAE_layers, self.RAE_n_blocks, self.IB_btnk_dim, self.RAE_kernel_size, dropout=0, use_bn=False)
        self.netRAE_LV = ResidualAE(self.RAE_layers, self.RAE_n_blocks, self.IB_btnk_dim, self.RAE_kernel_size, dropout=0, use_bn=False)
        self.netRAE_AV = ResidualAE(self.RAE_layers, self.RAE_n_blocks, self.IB_btnk_dim, self.RAE_kernel_size, dropout=0, use_bn=False)



    def get_network(self, self_type='l', layers=-1):
        if self_type in ['l', 'al', 'vl']:
            embed_dim, attn_dropout = self.common_dim, self.attn_dropout_l
        elif self_type in ['a', 'la', 'va']:
            embed_dim, attn_dropout = self.common_dim, self.attn_dropout_a
        elif self_type in ['v', 'lv', 'av']:
            embed_dim, attn_dropout = self.common_dim, self.attn_dropout_v
        elif self_type == 'l_mem':
            embed_dim, attn_dropout = 2 * self.common_dim, self.attn_dropout_mem
        elif self_type == 'a_mem':
            embed_dim, attn_dropout = 2 * self.common_dim, self.attn_dropout_mem
        elif self_type == 'v_mem':
            embed_dim, attn_dropout = 2 * self.common_dim, self.attn_dropout_mem
        else:
            raise ValueError("Unknown network type")

        # TODO: Replace with nn.TransformerEncoder
        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=0.0,
                                  res_dropout=0.0,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)

    def kl_loss(self, mu, logvar):
        kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        kl_mean = torch.mean(kl_div)
        return kl_mean

    def reparameterise(self, mu, logvar):
        epsilon = torch.randn_like(mu)
        return mu + epsilon * torch.exp(logvar / 2)
    
    def IB_translate(self, encoder_net, mlp_net, x_i, x_o, coeff):
        h = encoder_net(x_i)
        mu, logvar = h.chunk(2, dim=1)
        kl_loss = self.kl_loss(mu, logvar)
        btnk = self.reparameterise(mu, logvar)
        output = mlp_net(btnk)

        mse = self.MSE_loss(output, x_o)
        IB_loss = kl_loss + coeff * mse

        return btnk, IB_loss

    def forward(self, text, audio, vision, mask_matrix=None, generation_stage=False):
        _text_z = self.text_tokenizer(text,
                                        add_special_tokens=True,
                                        max_length=self.args.max_text_length, padding='max_length', truncation=True,
                                        return_tensors='pt').to(self.device)
        text = self.text_model(**_text_z).last_hidden_state

        text, audio, vision = self.alignNet(text, audio, vision)

        x_l = text.transpose(1, 2) # (N, d_l, L)
        x_a = audio.transpose(1, 2)
        x_v = vision.transpose(1, 2)

        if self.training:
            ################## token-level IB to build informative sapce ##################
            btnk_l, IB_l1 = self.IB_translate(self.encoder_l, self.mlp_l, x_l, x_l, self.p_eta)
            btnk_l, IB_l2 = self.IB_translate(self.encoder_l, self.mlp_a, x_l, x_a, self.p_eta)
            btnk_l, IB_l3 = self.IB_translate(self.encoder_l, self.mlp_v, x_l, x_v, self.p_eta)
            btnk_a, IB_a1 = self.IB_translate(self.encoder_a, self.mlp_a, x_a, x_a, self.p_eta)
            btnk_a, IB_a2 = self.IB_translate(self.encoder_a, self.mlp_l, x_a, x_l, self.p_eta)
            btnk_a, IB_a3 = self.IB_translate(self.encoder_a, self.mlp_v, x_a, x_v, self.p_eta)
            btnk_v, IB_v1 = self.IB_translate(self.encoder_v, self.mlp_v, x_v, x_v, self.p_eta)
            btnk_v, IB_v2 = self.IB_translate(self.encoder_v, self.mlp_l, x_v, x_l, self.p_eta)
            btnk_v, IB_v3 = self.IB_translate(self.encoder_v, self.mlp_a, x_v, x_a, self.p_eta)

            # MI loss for information bottleneck 
            # IB_loss = 1/3 * (IB_l1 + IB_a1 + IB_v1) # ablation no cyclic interaction
            IB_loss = 1/9 * (IB_l1 + IB_l2 + IB_l3 + IB_a1 + IB_a2 + IB_a3 + IB_v1 + IB_v2 + IB_v3) 

            ################## label-level IB to build informative sapce ##################
            output_l = self.out_layer_l(btnk_l.mean(dim=-1))
            output_a = self.out_layer_a(btnk_a.mean(dim=-1))
            output_v = self.out_layer_v(btnk_v.mean(dim=-1))
            
            TRANSLATE_loss = torch.tensor(0., dtype=torch.float32).to(self.device)

            if generation_stage:
                ################## Cyclic Forward Translation by CRA ##################
                btnk_a_gen_l, _ = self.netRAE_LA(btnk_a) # A -> L_gen
                btnk_v_gen_l, _ = self.netRAE_LV(btnk_v) # V -> L_gen
                # btnk_av_gen_l, _ = self.netRAE_LAV(btnk_a+btnk_v) # A + V -> L_gen
                btnk_av_gen_l = btnk_a_gen_l + btnk_v_gen_l # Gaussian Distribution additive

                btnk_l_gen_a, _ = self.netRAE_LA(btnk_l) # L -> A_gen
                btnk_v_gen_a, _ = self.netRAE_AV(btnk_v) # V -> A_gen
                # btnk_lv_gen_a, _ = self.netRAE_LAV(btnk_l+btnk_v) # L + V -> A_gen
                btnk_lv_gen_a = btnk_l_gen_a + btnk_v_gen_a

                btnk_l_gen_v, _ = self.netRAE_LV(btnk_l) # L -> V_gen
                btnk_a_gen_v, _ = self.netRAE_AV(btnk_a) # A -> V_gen
                # btnk_al_gen_v, _ = self.netRAE_LAV(btnk_l+btnk_a) # L + A -> V_gen
                btnk_al_gen_v = btnk_l_gen_v + btnk_a_gen_v

                TRANSLATE_loss = 1/9 * (self.MSE_loss(btnk_a_gen_l, btnk_l) + self.MSE_loss(btnk_v_gen_l, btnk_l) + self.MSE_loss(btnk_av_gen_l, btnk_l) + self.MSE_loss(btnk_l_gen_a, btnk_a) + self.MSE_loss(btnk_v_gen_a, btnk_a) + self.MSE_loss(btnk_lv_gen_a, btnk_a) + self.MSE_loss(btnk_l_gen_v, btnk_v) + self.MSE_loss(btnk_a_gen_v, btnk_v) + self.MSE_loss(btnk_al_gen_v, btnk_v))
    
                if self.use_cycle_reverse:
                    ################## Cyclic Reverse Translation by CRA ##################
                    btnk_a_gen_l_rev, _ = self.netRAE_LA(btnk_l_gen_a) # A_gen -> L_rev
                    btnk_v_gen_l_rev, _ = self.netRAE_LV(btnk_l_gen_v) # V_gen -> L_rev
                    # btnk_av_gen_l_rev, _ = self.netRAE_LAV(btnk_l_gen_a+btnk_l_gen_v) # A_gen + V_gen -> L_rev
                    btnk_av_gen_l_rev = btnk_a_gen_l_rev + btnk_v_gen_l_rev

                    btnk_l_gen_a_rev, _ = self.netRAE_LA(btnk_a_gen_l) # L_gen -> A_rev
                    btnk_v_gen_a_rev, _ = self.netRAE_AV(btnk_a_gen_v) # V_gen -> A_rev
                    # btnk_lv_gen_a_rev, _ = self.netRAE_LAV(btnk_a_gen_l+btnk_a_gen_v) # L_gen + V_gen -> A_rev
                    btnk_lv_gen_a_rev = btnk_l_gen_a_rev + btnk_v_gen_a_rev

                    btnk_l_gen_v_rev, _ = self.netRAE_LV(btnk_v_gen_l) # L_gen -> V_rev
                    btnk_a_gen_v_rev, _ = self.netRAE_AV(btnk_v_gen_a) # A_gen -> V_rev
                    # btnk_al_gen_v_rev, _ = self.netRAE_LAV(btnk_v_gen_l+btnk_v_gen_a) # L_gen + A_gen -> V_rev
                    btnk_al_gen_v_rev = btnk_l_gen_v_rev + btnk_a_gen_v_rev

                    TRANSLATE_loss += 1/9 * (self.MSE_loss(btnk_a_gen_l_rev, btnk_l) + self.MSE_loss(btnk_v_gen_l_rev, btnk_l) + self.MSE_loss(btnk_av_gen_l_rev, btnk_l) + self.MSE_loss(btnk_l_gen_a_rev, btnk_a) + self.MSE_loss(btnk_v_gen_a_rev, btnk_a) + self.MSE_loss(btnk_lv_gen_a_rev, btnk_a) + self.MSE_loss(btnk_l_gen_v_rev, btnk_v) + self.MSE_loss(btnk_a_gen_v_rev, btnk_v) + self.MSE_loss(btnk_al_gen_v_rev, btnk_v))

                    # Sample generated representation randomly for multimodal fusion
                    btnk_l = sample([btnk_a_gen_l, btnk_v_gen_l, btnk_av_gen_l, btnk_l], 1)[0]
                    btnk_a = sample([btnk_l_gen_a, btnk_v_gen_a, btnk_lv_gen_a, btnk_a], 1)[0]
                    btnk_v = sample([btnk_a_gen_v, btnk_l_gen_v, btnk_al_gen_v, btnk_v], 1)[0]

        else:
            
            # token-level IB to build informative sapce
            btnk_l = self.reparameterise(*self.encoder_l(x_l).chunk(2, dim=1))
            btnk_a = self.reparameterise(*self.encoder_a(x_a).chunk(2, dim=1))
            btnk_v = self.reparameterise(*self.encoder_v(x_v).chunk(2, dim=1))
            
            IB_loss = torch.tensor(0., dtype=torch.float32).to(self.device)
            TRANSLATE_loss = torch.tensor(0., dtype=torch.float32).to(self.device)

            # Cyclic Translation by CRA
            btnk_a_gen_l, _ = self.netRAE_LA(btnk_a) # A -> L_gen
            btnk_v_gen_l, _ = self.netRAE_LV(btnk_v) # V -> L_gen
            # btnk_av_gen_l, _ = self.netRAE_LAV(btnk_a+btnk_v) # A + V -> L_gen
            btnk_av_gen_l = btnk_a_gen_l + btnk_v_gen_l # Gaussian Distribution additive

            btnk_l_gen_a, _ = self.netRAE_LA(btnk_l) # L -> A_gen
            btnk_v_gen_a, _ = self.netRAE_AV(btnk_v) # V -> A_gen
            # btnk_lv_gen_a, _ = self.netRAE_LAV(btnk_l+btnk_v) # L + V -> A_gen
            btnk_lv_gen_a = btnk_l_gen_a + btnk_v_gen_a

            btnk_l_gen_v, _ = self.netRAE_LV(btnk_l) # L -> V_gen
            btnk_a_gen_v, _ = self.netRAE_AV(btnk_a) # A -> V_gen
            # btnk_al_gen_v, _ = self.netRAE_LAV(btnk_l+btnk_a) # L + A -> V_gen
            btnk_al_gen_v = btnk_l_gen_v + btnk_a_gen_v
            
            assert mask_matrix != None, "Error: no mask matrix setting!!!"
            mask_l = mask_matrix[:, 0]  # [B]
            mask_a = mask_matrix[:, 1]  # [B]
            mask_v = mask_matrix[:, 2]  # [B]

            mask_text_av = ((mask_l == 0) & (mask_a == 1) & (mask_v == 1)).unsqueeze(-1).unsqueeze(-1)  # audio 和 vision 均可用
            mask_text_a  = ((mask_l == 0) & (mask_a == 1) & (mask_v == 0)).unsqueeze(-1).unsqueeze(-1)  # 仅 audio 可用
            mask_text_v  = ((mask_l == 0) & (mask_a == 0) & (mask_v == 1)).unsqueeze(-1).unsqueeze(-1)  # 仅 vision 可用
            btnk_l = torch.where(mask_text_av, btnk_av_gen_l, btnk_l)
            btnk_l = torch.where(mask_text_a,  btnk_a_gen_l,  btnk_l)
            btnk_l = torch.where(mask_text_v,  btnk_v_gen_l,  btnk_l)

            mask_audio_lv = ((mask_a == 0) & (mask_l == 1) & (mask_v == 1)).unsqueeze(-1).unsqueeze(-1)  # text 和 vision 均可用
            mask_audio_l  = ((mask_a == 0) & (mask_l == 1) & (mask_v == 0)).unsqueeze(-1).unsqueeze(-1)  # 仅 text 可用
            mask_audio_v  = ((mask_a == 0) & (mask_l == 0) & (mask_v == 1)).unsqueeze(-1).unsqueeze(-1)  # 仅 vision 可用
            btnk_a = torch.where(mask_audio_lv, btnk_lv_gen_a, btnk_a)
            btnk_a = torch.where(mask_audio_l,  btnk_l_gen_a,  btnk_a)
            btnk_a = torch.where(mask_audio_v,  btnk_v_gen_a,  btnk_a)

            mask_vision_la = ((mask_v == 0) & (mask_l == 1) & (mask_a == 1)).unsqueeze(-1).unsqueeze(-1)  # text 和 audio 均可用
            mask_vision_l  = ((mask_v == 0) & (mask_l == 1) & (mask_a == 0)).unsqueeze(-1).unsqueeze(-1)  # 仅 text 可用
            mask_vision_a  = ((mask_v == 0) & (mask_l == 0) & (mask_a == 1)).unsqueeze(-1).unsqueeze(-1)  # 仅 audio 可用
            btnk_v = torch.where(mask_vision_la, btnk_al_gen_v, btnk_v)
            btnk_v = torch.where(mask_vision_l,  btnk_l_gen_v,  btnk_v)
            btnk_v = torch.where(mask_vision_a,  btnk_a_gen_v,  btnk_v)

        # Project the textual/visual/audio features into common dimension & process temporal information
        btnk_l = self.proj_l(btnk_l) # Dimension (N, d_l, L)
        btnk_a = self.proj_a(btnk_a)
        btnk_v = self.proj_v(btnk_v)

        ''' Multimodal Fusion based on MulT '''
        proj_x_a = btnk_a.permute(2, 0, 1)  # Dimension (L, N, d_l)
        proj_x_v = btnk_v.permute(2, 0, 1)
        proj_x_l = btnk_l.permute(2, 0, 1)
        # (V,A) --> L
        h_l_with_as = self.trans_l_with_a(proj_x_l, proj_x_a, proj_x_a) # Dimension (L, N, d_l)
        h_l_with_vs = self.trans_l_with_v(proj_x_l, proj_x_v, proj_x_v)
        h_ls = torch.cat([h_l_with_as, h_l_with_vs], dim=-1)
        h_ls = self.trans_l_mem(h_ls)
        if type(h_ls) == tuple:
            h_ls = h_ls[0]
        last_h_l = h_ls.mean(dim=0) #= h_ls[-1]  # Take the last output for prediction

        # (L,V) --> A
        h_a_with_ls = self.trans_a_with_l(proj_x_a, proj_x_l, proj_x_l)
        h_a_with_vs = self.trans_a_with_v(proj_x_a, proj_x_v, proj_x_v)
        h_as = torch.cat([h_a_with_ls, h_a_with_vs], dim=-1)
        h_as = self.trans_a_mem(h_as)
        if type(h_as) == tuple:
            h_as = h_as[0]
        last_h_a = h_as.mean(dim=0) #= h_as[-1]

        # (L,A) --> V
        h_v_with_ls = self.trans_v_with_l(proj_x_v, proj_x_l, proj_x_l)
        h_v_with_as = self.trans_v_with_a(proj_x_v, proj_x_a, proj_x_a)
        h_vs = torch.cat([h_v_with_ls, h_v_with_as], dim=-1)
        h_vs = self.trans_v_mem(h_vs)
        if type(h_vs) == tuple:
            h_vs = h_vs[0]
        last_h_v = h_vs.mean(dim=0) #= h_vs[-1]

        # Final Representation
        last_hs = torch.cat([last_h_l, last_h_v, last_h_a], dim=-1) # Dimension (L, d_l)
        last_hs_proj = self.proj2(
            F.dropout(F.relu(self.proj1(last_hs), inplace=True), p=self.output_dropout, training=self.training))
        last_hs_proj += last_hs

        output = self.out_layer(last_hs_proj)

        if self.training:
            res = {
                'IB_loss': IB_loss,
                'TRANSLATE_loss': TRANSLATE_loss,
                'L': output_l,
                'A': output_a,
                'V': output_v,
                'M': output
            }
        else:
            res = {
                'M': output,
            }

        return res
