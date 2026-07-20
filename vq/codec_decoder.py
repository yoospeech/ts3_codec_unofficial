import numpy as np
import torch
import torch.nn as nn
from .residual_vq import ResidualVQ
from .module import WNConv1d, DecoderBlock, ResLSTM
from .alias_free_torch import *
from . import activations
from .llama_mini import ModelArgs, Transformer

def init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        nn.init.constant_(m.bias, 0)

class CodecDecoder(nn.Module):
    def __init__(self,
                 in_channels=1024,
                 upsample_initial_channel=1536,
                 ngf=48,
                 use_rnn=True,
                 rnn_bidirectional=False,
                 rnn_num_layers=2,
                 up_ratios=(5, 5, 2, 2, 2),
                 dilations=(1, 3, 9),
                 vq_num_quantizers=1,
                 vq_dim=1024,
                 vq_commit_weight=0.25,
                 vq_weight_init=False,
                 vq_full_commit_loss=False,
                 codebook_size=8192,
                 codebook_dim=8,
                 transformer_only=False,
                 transformer_num_layers=8,
                 transformer_num_heads=16,
                 transformer_ffn_dim=4096,
                 transformer_dropout=0.1,
                 attention_window=0,
                 causal_attention=True,
                 frame_hidden_dim=768,
                ):
        super().__init__()
        self.hop_length = np.prod(up_ratios)
        self.ngf = ngf
        self.up_ratios = up_ratios
        self.transformer_only = transformer_only
        self.attention_window = attention_window
        self.causal_attention = causal_attention

        self.quantizer = ResidualVQ(
            num_quantizers=vq_num_quantizers,
            dim=vq_dim,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            threshold_ema_dead_code=2,
            commitment=vq_commit_weight,
            weight_init=vq_weight_init,
            full_commit_loss=vq_full_commit_loss,
        )

        if self.transformer_only:
            transformer_args = ModelArgs(
                dim=in_channels,
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
            self.final_norm = nn.LayerNorm(in_channels)
            self.linear_1 = nn.Linear(in_channels, frame_hidden_dim, bias=True)
            self.linear_2 = nn.Linear(frame_hidden_dim, self.hop_length, bias=False)
            self.reset_parameters()
            return

        channels = upsample_initial_channel
        layers = [WNConv1d(in_channels, channels, kernel_size=7, padding=3)]

        if use_rnn:
            layers += [
                ResLSTM(channels,
                        num_layers=rnn_num_layers,
                        bidirectional=rnn_bidirectional
                    )
            ]

        for i, stride in enumerate(up_ratios):
            input_dim = channels // 2**i
            output_dim = channels // 2 ** (i + 1)
            layers += [DecoderBlock(input_dim, output_dim, stride, dilations)]

        layers += [
            Activation1d(activation=activations.SnakeBeta(output_dim, alpha_logscale=True)),
            WNConv1d(output_dim, 1, kernel_size=7, padding=3),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*layers)

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

    def _decode(self, x, lengths=None):
        if not self.transformer_only:
            return self.model(x)
        x = x.transpose(1, 2)
        attn_mask = self._build_attn_mask(x.shape[1], x.device, x.dtype)
        key_padding_mask = self._build_key_padding_mask(lengths, x.shape[1], x.device)
        x = self.transformer(x, attn_mask=attn_mask, src_key_padding_mask=key_padding_mask)
        x = self.final_norm(x)
        x = self.linear_1(x)
        x = self.linear_2(x)
        bsz = x.shape[0]
        x = x.reshape(bsz, -1)
        #x = torch.clip(x, min=-1.0, max=1.0)
        x = x.unsqueeze(1)
        return x

    def forward(self, x, vq=True, lengths=None):
        if vq is True:
            input_dtype = x.dtype
            with torch.autocast(device_type=x.device.type, enabled=False):
                x, q, commit_loss, utilization = self.quantizer(x.float())
            x = x.to(dtype=input_dtype)
            return x, q, commit_loss, utilization
        x = self._decode(x, lengths=lengths)
        return x

    def vq2emb(self, vq):
        self.quantizer = self.quantizer.eval()
        x = self.quantizer.vq2emb(vq)
        return x

    def get_emb(self):
        self.quantizer = self.quantizer.eval()
        embs = self.quantizer.get_emb()
        return embs

    def inference_vq(self, vq, lengths=None):
        x = vq[None, :, :]
        x = self._decode(x, lengths=lengths)
        return x

    def inference_0(self, x, lengths=None):
        x, q, loss, perp = self.quantizer(x)
        x = self._decode(x, lengths=lengths)
        return x, None

    def inference(self, x, lengths=None):
        x = self._decode(x, lengths=lengths)
        return x, None

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""

        def _remove_weight_norm(m):
            try:
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""

        def _apply_weight_norm(m):
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                torch.nn.utils.weight_norm(m)

        self.apply(_apply_weight_norm)

    def reset_parameters(self):
        self.apply(init_weights)
