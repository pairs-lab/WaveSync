# WaveSync
<div align="center">
<h1>🌊 WaveSync</h1>
<p><strong>Constrained Wavefront Optimization for Synchronized Co-Speech Gestures in Humanoid Robots</strong></p>
<p>
  <a href="https://arxiv.org/abs/2606._">
    <img src="https://img.shields.io/badge/arXiv-2606.___-b31b1b?style=flat-square&logo=arxiv" alt="arXiv"/>
  </a>
  <img src="https://img.shields.io/badge/Python-3.8%2B-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="License"/>
</p>

<p>
  <video src="https://github.com/user-attachments/assets/10836066-5fe5-47af-9907-98259508199c" width="90%" autoplay loop muted playsinline></video>
</p>


## Installation
Install the Python dependencies:
```bash
pip install -r requirements.txt
```

## Dataset & Model Setup

Before running the simulation, you need to set up the required data and model weights.

1. Download the `data` and `out/models` folders from [Google Drive](https://drive.google.com/drive/folders/1H9Bt5Vaat80koK4s8-PDnQ5K79fKh3ND?usp=sharing).
2. Place these folders in the root directory according to the structure shown below:

![Directory Structure](docs/structure.png)

## Usage

The core simulation is executed via `execute.py`. You can trigger predefined scenes from the `scene/` directory.

```bash
# Run scenes
python execute.py -s scene1.json
python execute.py -s scene2.json
python execute.py -s scene3.json
python execute.py -s scene4.json
python execute.py -s scene5.json
```
## Citation

```bibtex
@article{viet2026wavesync,
  title={ WaveSync: Constrained Wavefront Optimization for Synchronized Co-Speech  Gestures in Humanoid Robots},
  author={Thang Tran Viet, Thanh Nguyen Canh, Gia Huy Uong, Phuc Van Dinh, Tan Viet Tuyen Nguyen, Xiem HoangVan, and Nak Young Chong},
  journal={arXiv preprint arXiv:2606._},
  year={2026}
}
```
