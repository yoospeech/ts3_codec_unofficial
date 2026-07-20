import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from .module import WNConv1d, EncoderBlock, ResLSTM
from .alias_free_torch import *
from . import activations
from .llama_mini import ModelArgs, Transformer

def init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        nn.init.constant_(m.bias, 0)

class CodecEncoder(nn.Module):
    def __init__(self,
                ngf=48,
                use_rnn=True,
                rnn_bidirectional=False,
                rnn_num_layers=2,
                up_ratios=(2, 2, 2, 5, 5),
                dilations=(1, 3, 9),
                out_channels=1024,
                transformer_only=False,
                transformer_num_layers=8,
                transformer_num_heads=16,
                transformer_ffn_dim=4096,
                transformer_dropout=0.1,
                attention_window=0,
                causal_attention=True,
                frame_hidden_dim=768):
        super().__init__()
        self.hop_length = np.prod(up_ratios)
        self.ngf = ngf
        self.up_ratios = up_ratios
        self.transformer_only = transformer_only
        self.attention_window = attention_window
        self.causal_attention = causal_attention
        self.out_channels = out_channels

        if self.transformer_only:
            self.frame_in = nn.Linear(self.hop_length, frame_hidden_dim, bias=False)
            self.feat_in = nn.Linear(frame_hidden_dim, out_channels, bias=True)
            transformer_args = ModelArgs(
                dim=out_channels,
                n_layers=transformer_num_layers,
                n_heads=transformer_num_heads,
                n_kv_heads=transformer_num_heads,
                hidden_dim=transformer_ffn_dim,
                vocab_size=8,
                dropout=transformer_dropout,
                max_seq_len=max(2048, attention_window),
                relative_attention_window=max(1, attention_window),
                use_relative_attention=True,
            )
            self.transformer = Transformer(transformer_args)
            self.final_norm = nn.LayerNorm(out_channels)
            self.enc_dim = out_channels
            self.reset_parameters()
            return

        # Create first convolution
        d_model = ngf
        self.block = [WNConv1d(1, d_model, kernel_size=7, padding=3)]

        # Create EncoderBlocks that double channels as they downsample by `stride`
        for i, stride in enumerate(up_ratios):
            d_model *= 2
            self.block += [EncoderBlock(d_model, stride=stride, dilations=dilations)]
        # RNN
        if use_rnn:
            self.block += [
                ResLSTM(d_model,
                        num_layers=rnn_num_layers,
                        bidirectional=rnn_bidirectional
                    )
            ]
        # Create last convolution
        self.block += [
            Activation1d(activation=activations.SnakeBeta(d_model, alpha_logscale=True)),
            WNConv1d(d_model, out_channels, kernel_size=3, padding=1),
        ]

        # Wrap black into nn.Sequential
        self.block = nn.Sequential(*self.block)
        self.enc_dim = d_model

        self.reset_parameters()

    def _build_attn_mask(self, seq_len, device, dtype):
        if self.attention_window <= 0 and not self.causal_attention:
            return None
        q = torch.arange(seq_len, device=device)[:, None]
        k = torch.arange(seq_len, device=device)[None, :]
        invalid = torch.zeros((seq_len, seq_len), dtype=torch.bool, device=device)
        if self.causal_attention:
            invalid |= k > q
        if self.attention_window > 0:
            invalid |= (q - k) >= self.attention_window
        mask = torch.zeros((seq_len, seq_len), device=device, dtype=dtype)
        mask = mask.masked_fill(invalid, float('-inf'))
        return mask

    def _build_key_padding_mask(self, lengths, frame_num, device):
        if lengths is None:
            return None
        if not torch.is_tensor(lengths):
            lengths = torch.LongTensor(lengths)
        lengths = lengths.to(device=device, dtype=torch.long)
        frame_lengths = torch.div(lengths + self.hop_length - 1, self.hop_length, rounding_mode='floor')
        frame_lengths = frame_lengths.clamp(min=0, max=frame_num)
        frame_index = torch.arange(frame_num, device=device).unsqueeze(0)
        key_padding_mask = frame_index >= frame_lengths.unsqueeze(1)
        if not torch.any(key_padding_mask):
            return None
        return key_padding_mask

    def _forward_transformer(self, x, lengths=None):
        bsz, _, time_len = x.shape
        pad_len = (self.hop_length - (time_len % self.hop_length)) % self.hop_length
        if pad_len > 0:
            x = F.pad(x, (0, pad_len))
        frame_num = x.shape[-1] // self.hop_length
        x = x.squeeze(1).reshape(bsz, frame_num, self.hop_length)
        x = self.frame_in(x)
        x = self.feat_in(x)
        attn_mask = self._build_attn_mask(x.shape[1], x.device, x.dtype)
        key_padding_mask = self._build_key_padding_mask(lengths, frame_num, x.device)
        x = self.transformer(x, attn_mask=attn_mask, src_key_padding_mask=key_padding_mask)
        x = self.final_norm(x)
        x = x.transpose(1, 2)
        return x

    def forward(self, x, lengths=None):
        if self.transformer_only:
            return self._forward_transformer(x, lengths=lengths)
        out = self.block(x)
        return out

    def inference(self, x, lengths=None):
        if self.transformer_only:
            return self._forward_transformer(x, lengths=lengths)
        return self.block(x)

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""

        def _remove_weight_norm(m):
            try:
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""

        def _apply_weight_norm(m):
            if isinstance(m, nn.Conv1d):
                torch.nn.utils.weight_norm(m)

        self.apply(_apply_weight_norm)

    def reset_parameters(self):
        self.apply(init_weights)