# TS3Codec (Unofficial)

Unofficial training and inference implementation of
**[TS3-Codec: Transformer-Based Simple Streaming Single Codec](https://arxiv.org/abs/2411.18803)**
for 16 kHz speech. This repository is derived from the
[BigCodec](https://github.com/Aria-K-Alethia/BigCodec) implementation and keeps
the original MIT license and attribution.

> This is an unofficial research implementation and is not affiliated with the
> authors of the paper.

## Features

- 16 kHz waveform codec training with PyTorch Lightning
- Transformer-based causal encoder and decoder
- Low-dimensional residual vector quantization
- Multi-period and multi-resolution STFT adversarial training
- File-list and JSON-manifest dataset loading
- Checkpoint resume and standalone waveform reconstruction

## Installation

Python 3.9 or a compatible environment is recommended.

```bash
git clone <repository-url>
cd BigCodec_trn_16khz_repo
pip install -r requirements.txt
```

## Dataset manifest

Dataset manifests are intentionally excluded because they contain local paths.
The loader accepts either a text file list or a JSON manifest. A JSON manifest
can contain entries such as:

```json
[
  {"audio_filepath": "/absolute/path/to/audio.wav"}
]
```

Set `TRAIN_FILELIST`, `VAL_FILELIST`, and `TEST_FILELIST` to your own manifests.
You can create a JSON manifest from a text file containing one WAV path per line:

```bash
python make_json.py wav_paths.txt manifest.json
```

## Training data

The released pretrained model was trained on a combination of:

- [LibriSpeech](https://ieeexplore.ieee.org/document/7178964), an English
  read-speech corpus
- [KsponSpeech](https://www.mdpi.com/2076-3417/10/19/6936), a Korean
  spontaneous-speech corpus

Dataset files and manifests are not redistributed in this repository. Users
must obtain each dataset separately and comply with its applicable terms.

## Training

The shell entry point contains the 16 kHz and model overrides used for training:

```bash
TRAIN_FILELIST=train.json bash ./train.sh dataset.random_val_ratio=0.001 dataset.random_val_seed=42
```

Resume training with a Lightning checkpoint:

```bash
./train.sh +resume_ckpt=/path/to/last.ckpt
```

Additional Hydra overrides may be appended to either command.

## Pretrained checkpoint

The pretrained Lightning checkpoint is hosted in the
[TS3Codec unofficial Hugging Face model repository](https://huggingface.co/youspeech/ts3_codec_unofficial)
rather than Git because of its size. Download the checkpoint and its checksum:

```bash
hf download \
  youspeech/ts3_codec_unofficial \
  ts3codec_16khz.ckpt \
  SHA256SUMS \
  --local-dir checkpoints

cd checkpoints
sha256sum -c SHA256SUMS
cd ..
```

Then run inference with:

```bash
CKPT=checkpoints/ts3codec_16khz.ckpt \
INPUT_DIR=/path/to/input_wavs \
OUTPUT_DIR=./recon_wavs \
./inference.sh
```

## Audio samples

Example reconstruction for LibriTTS-R utterance `2078_142845_000085_000003`:

- [Original (24 kHz)](samples/2078_142845_000085_000003_original_24khz.wav)
- [TS3Codec reconstruction (16 kHz)](samples/2078_142845_000085_000003_reconstructed_16khz.wav)

The source and reconstruction have different sample rates. See
[samples/README.md](samples/README.md) for provenance and licensing information.

## Training logs

The released pretrained checkpoint was trained for approximately **81,500
steps** over about **10 days** on an **NVIDIA DGX Spark**. Representative
TensorBoard logs are published under `training_logs/`. View them with:

```bash
tensorboard --logdir training_logs
```

## Acknowledgements

This project builds on the official BigCodec codebase:

- [BigCodec](https://github.com/Aria-K-Alethia/BigCodec) provides the original
  neural codec training framework.
- [llama2.c](https://github.com/karpathy/llama2.c) by Andrej Karpathy was a
  major reference for the Transformer implementation in `vq/llama_mini.py`.

Third-party copyright and license notices are retained in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

```bibtex
@article{xin2024bigcodec,
  title={BigCodec: Pushing the Limits of Low-Bitrate Neural Speech Codec},
  author={Xin, Detai and Tan, Xu and Takamichi, Shinnosuke and Saruwatari, Hiroshi},
  journal={arXiv preprint arXiv:2409.05377},
  year={2024}
}
```

Please also cite TS3-Codec:

```bibtex
@article{wu2024ts3codec,
  title={TS3-Codec: Transformer-Based Simple Streaming Single Codec},
  author={Wu, Haibin and Kanda, Naoyuki and Eskimez, Sefik Emre and Li, Jinyu},
  journal={arXiv preprint arXiv:2411.18803},
  year={2024}
}
```

### Dataset citations

```bibtex
@inproceedings{panayotov2015librispeech,
  title={LibriSpeech: An ASR Corpus Based on Public Domain Audio Books},
  author={Panayotov, Vassil and Chen, Guoguo and Povey, Daniel and Khudanpur, Sanjeev},
  booktitle={2015 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={5206--5210},
  year={2015},
  doi={10.1109/ICASSP.2015.7178964}
}

@article{bang2020ksponspeech,
  title={KsponSpeech: Korean Spontaneous Speech Corpus for Automatic Speech Recognition},
  author={Bang, Jeong-Uk and Yun, Seung and Kim, Seung-Hi and Choi, Mu-Yeol and Lee, Min-Kyu and Kim, Yeo-Jeong and Kim, Dong-Hyun and Park, Jun and Lee, Young-Jik and Kim, Sang-Hun},
  journal={Applied Sciences},
  volume={10},
  number={19},
  pages={6936},
  year={2020},
  doi={10.3390/app10196936}
}
```

## License

MIT. See [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
