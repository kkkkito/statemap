# StateMap

## Overview

This repository contains code used to analyze data and generate figures related to **StateMap**.

StateMap is also available as a web application:
https://lexas.f.u-tokyo.ac.jp/statemap/

Notebooks used for figure generation are located in the `notebook/` directory.

## Requirements

```text
numpy==2.0.2
pandas==2.3.3
scipy==1.13.1
scikit-learn==1.6.1
matplotlib==3.9.4
seaborn==0.13.2
trimap==1.1.5
umap-learn==0.5.11
networkx==3.2.1
tqdm==4.67.3
```

## Installation and usage

1. Clone this repository:

```bash
git clone https://github.com/kkkkito/statemap.git
cd statemap
```

2. Create a `data/` directory and download the required data from the following link:

[Google Drive - StateMap Data](https://drive.google.com/drive/folders/1VJcXN3_FH6hemraymy6BAhYu1g9wAVR6?usp=sharing)

3. Place the downloaded files in the `data/` directory.

4. Run the main.ipynb
