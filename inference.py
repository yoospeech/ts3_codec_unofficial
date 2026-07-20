import os
import hydra
import librosa
import torch
import soundfile as sf
from glob import glob
from tqdm import tqdm
from os.path import basename, join
from vq.codec_encoder import CodecEncoder
from vq.codec_decoder import CodecDecoder
from time import time

def _extract_module_state_dict(ckpt, module_name):
    if module_name in ckpt and isinstance(ckpt[module_name], dict):
        return ckpt[module_name]
    state_dict = ckpt.get('state_dict', None)
    if state_dict is None:
        raise KeyError(f'Cannot find `{module_name}` state dict in checkpoint.')
    prefix = f'model.{module_name}.'
    module_state_dict = {
        key[len(prefix):]: value
        for key, value in state_dict.items()
        if key.startswith(prefix)
    }
    if len(module_state_dict) == 0:
        raise KeyError(f'Cannot find prefix `{prefix}` in checkpoint state_dict.')
    return module_state_dict


def _build_models(cfg, ckpt, device):
    enccfg = cfg.model.codec_encoder
    deccfg = cfg.model.codec_decoder

    encoder = CodecEncoder(
        ngf=enccfg.ngf,
        use_rnn=enccfg.use_rnn,
        rnn_bidirectional=enccfg.rnn_bidirectional,
        rnn_num_layers=enccfg.rnn_num_layers,
        up_ratios=enccfg.up_ratios,
        dilations=enccfg.dilations,
        out_channels=enccfg.out_channels,
        transformer_only=enccfg.get('transformer_only', False),
        transformer_num_layers=enccfg.get('transformer_num_layers', 8),
        transformer_num_heads=enccfg.get('transformer_num_heads', 16),
        transformer_ffn_dim=enccfg.get('transformer_ffn_dim', 4096),
        transformer_dropout=enccfg.get('transformer_dropout', 0.1),
        attention_window=enccfg.get('attention_window', 0),
        causal_attention=enccfg.get('causal_attention', True),
        frame_hidden_dim=enccfg.get('frame_hidden_dim', 768),
    )
    decoder = CodecDecoder(
        in_channels=deccfg.in_channels,
        upsample_initial_channel=deccfg.upsample_initial_channel,
        ngf=deccfg.ngf,
        use_rnn=deccfg.use_rnn,
        rnn_bidirectional=deccfg.rnn_bidirectional,
        rnn_num_layers=deccfg.rnn_num_layers,
        up_ratios=deccfg.up_ratios,
        dilations=deccfg.dilations,
        vq_num_quantizers=deccfg.vq_num_quantizers,
        vq_dim=deccfg.vq_dim,
        vq_commit_weight=deccfg.vq_commit_weight,
        vq_weight_init=deccfg.vq_weight_init,
        vq_full_commit_loss=deccfg.vq_full_commit_loss,
        codebook_size=deccfg.codebook_size,
        codebook_dim=deccfg.codebook_dim,
        transformer_only=deccfg.get('transformer_only', False),
        transformer_num_layers=deccfg.get('transformer_num_layers', 8),
        transformer_num_heads=deccfg.get('transformer_num_heads', 16),
        transformer_ffn_dim=deccfg.get('transformer_ffn_dim', 4096),
        transformer_dropout=deccfg.get('transformer_dropout', 0.1),
        attention_window=deccfg.get('attention_window', 0),
        causal_attention=deccfg.get('causal_attention', True),
        frame_hidden_dim=deccfg.get('frame_hidden_dim', 768),
    )

    encoder_state_dict = _extract_module_state_dict(ckpt, 'CodecEnc')
    decoder_state_dict = _extract_module_state_dict(ckpt, 'generator')
    encoder.load_state_dict(encoder_state_dict)
    decoder.load_state_dict(decoder_state_dict)
    encoder = encoder.eval().to(device)
    decoder = decoder.eval().to(device)
    return encoder, decoder


@hydra.main(config_path='config', config_name='default', version_base=None)
def main(cfg):
    if cfg.ckpt is None or cfg.input_dir is None or cfg.output_dir is None:
        raise ValueError('Please set `ckpt`, `input_dir`, and `output_dir`.')

    sr = cfg.preprocess.audio.sr
    hop_length = cfg.preprocess.stft.hop_length
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'Load codec ckpt from {cfg.ckpt}')
    ckpt = torch.load(cfg.ckpt, map_location='cpu', weights_only=False)
    encoder, decoder = _build_models(cfg, ckpt, device)

    wav_dir = cfg.output_dir
    os.makedirs(wav_dir, exist_ok=True)
    wav_paths = glob(join(cfg.input_dir, '*.wav'))
    print(f'Found {len(wav_paths)} wavs in {cfg.input_dir}')

    st = time()
    for wav_path in tqdm(wav_paths):
        target_wav_path = join(wav_dir, basename(wav_path))
        wav = librosa.load(wav_path, sr=sr)[0]
        wav = torch.from_numpy(wav).unsqueeze(0).to(device)
        lengths = torch.LongTensor([wav.shape[1]]).to(device)
        pad_len = (hop_length - (wav.shape[1] % hop_length)) % hop_length
        wav = torch.nn.functional.pad(wav, (0, pad_len))
        with torch.no_grad():
            vq_emb = encoder(wav.unsqueeze(1), lengths=lengths)
            vq_post_emb, vq_code, _, _ = decoder(vq_emb, vq=True)
            recon = decoder(vq_post_emb, vq=False, lengths=lengths).squeeze(1).squeeze(0)
            recon = recon[:lengths[0]].detach().cpu().numpy()
        sf.write(target_wav_path, recon, sr)
    et = time()
    print(f'Inference ends, time: {(et-st)/60:.2f} mins')


if __name__ == '__main__':
    main()
