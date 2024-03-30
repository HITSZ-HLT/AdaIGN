# AdaIGN

> The official implementation for the conference of the AAAI 2024 paper *Adaptive Graph Learning for Multimodal Conversational Emotion Detection*.

<img src="https://img.shields.io/badge/Venue-AAAI--24-blue" alt="venue"/> <img src="https://img.shields.io/badge/Status-Accepted-success" alt="status"/> <img src="https://img.shields.io/badge/Issues-Welcome-red">

## Requirements
* Python 3.7.12
* PyTorch 1.12.0
* Transformers 4.27.4
* CUDA 10.1

## Preparation
Download [**features**](https://drive.google.com/file/d/1oUa1RMJvj8F-YOdG9Rlovi94jI_UDPIw/view?usp=drive_link) and save them in ./.

Download [**pseudo labels**](https://drive.google.com/file/d/1WrwtlWIVY_eziDYbDnm0cd6u7JouqZPB/view?usp=drive_link) and save them in ./.

## Training & Evaluation
You can train the models in the "code" folder with the following codes:

For IEMOCAP: ```python iemocap.py```

For MELD: ```python meld.py```

## Citation
If you find our work useful for your research, please kindly cite our paper as follows:

```
@inproceedings{tu2024adaptive,
  title={Adaptive Graph Learning for Multimodal Conversational Emotion Detection},
  author={Tu, Geng and Xie, Tian and Liang, Bin and Wang, Hongpeng and Xu, Ruifeng},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={17},
  pages={19089--19097},
  year={2024}
}
```

## Credits
The code of this repository partly relies on [MM-DFN](https://github.com/zerohd4869/MM-DFN) and I would like to show my sincere gratitude to the authors behind these contributions.

