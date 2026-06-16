# 🌊 WaveSync

**Constrained Wavefront Optimization for Synchronized Co-Speech Gestures in Humanoid Robots**

<a href="https://arxiv.org/abs/2606._">
  <img src="https://img.shields.io/badge/arXiv-2606.16600-b31b1b?style=flat-square&logo=arxiv" alt="arXiv"/>
</a>
<img src="https://img.shi


elds.io/badge/Python-3.8%2B-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python"/>
<img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="License"/>

<video src="https://github.com/user-attachments/assets/4f2ccf53-9ae7-4056-a6e0-af66e7cc60e6" width="90%" autoplay loop muted playsinline></video>

---

## 📖 Overview

**WaveSync** is a framework for generating temporally synchronized co-speech gestures in humanoid robots using constrained wavefront optimization. It enables natural, expressive gesture synthesis that aligns with the rhythm and semantics of speech in real time.

---

## ⚙️ Installation

Clone the repository and install the required Python dependencies:

```bash
git clone https://github.com/your-org/wavesync.git
cd wavesync
pip install -r requirements.txt
```

---

## 📦 Dataset & Model Setup

Before running any simulations, download the required data and pretrained model weights.

**Step 1 — Download assets from Google Drive:**

> 📁 [Download `data/` and `out/models/` here](https://drive.google.com/drive/folders/1H9Bt5Vaat80koK4s8-PDnQ5K79fKh3ND?usp=sharing)

**Step 2 — Place the folders in the root directory:**

```
wavesync/
├── data/               ← downloaded
├── out/
│   └── models/         ← downloaded
├── scene/
├── execute.py
└── requirements.txt
```

See [`docs/structure.png`](docs/structure.png) for a visual reference.

---

## 🚀 Usage

Run the core simulation via `execute.py` by passing a scene configuration file from the `scene/` directory.

```bash
python execute.py -s scene1.json
python execute.py -s scene2.json
python execute.py -s scene3.json
python execute.py -s scene4.json
python execute.py -s scene5.json
```

Each scene file defines a unique speech-gesture scenario. You can customize or create new scene configurations under `scene/`.

---

## 📁 Project Structure

```
wavesync/
├── data/               # Input speech and motion data
├── out/
│   └── models/         # Pretrained model weights
├── scene/              # Predefined scene configurations (JSON)
├── docs/               # Documentation and figures
├── execute.py          # Main entry point
└── requirements.txt    # Python dependencies
```

---

## 📄 Citation

If you find WaveSync useful in your research, please cite our paper:

```bibtex
@article{viet2026wavesync,
  title     = {WaveSync: Constrained Wavefront Optimization for Synchronized Co-Speech Gestures in Humanoid Robots},
  author    = {Thang Tran Viet and Thanh Nguyen Canh and Gia Huy Uong and Phuc Van Dinh and Tan Viet Tuyen Nguyen and Xiem HoangVan and Nak Young Chong},
  journal   = {arXiv preprint arXiv:2606.16600},
  year      = {2026}
}
```

