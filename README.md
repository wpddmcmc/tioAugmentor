# tioAugmentor

## Requirements

Tested on **Windows10**,**Anaconda3**, **Python==3.6.13** (should work for Linux and bash without Anaconda)

### Installation
For pip
```bash
pip install numpy==1.19.5
pip install matplotlib==3.3.4
pip install pillow==8.3.1
pip install torch==1.10.2
pip install torchvision==0.11.3
pip install torchio
pip install opencv-python==4.1.2.30
pip install opencv-contrib-python==4.1.2.30
```

For anaconda
```bash
conda install numpy==1.19.5
conda install matplotlib==3.3.4
conda install pillow==8.3.1
conda install torch==1.10.2
conda install torchvision==0.11.3
pip install torchio
pip install opencv-python==4.1.2.30
pip install opencv-contrib-python==4.1.2.30
```

## Usage
### Original data
The original image and labels should be saved in `./data/imgs/` and `./data/masks/` with `.png` format.
### Running
Parameters can be modified in  *line80-81*, realtime visualization based on PIL and augmentation times.

To run code
```bash
python ./augmentor.py
```

### Augmentation 
please check [TorchIO handbook](https://torchio.readthedocs.io/transforms/augmentation.html)

### Output
The generated data will be stored in `./newdata/imgs/` and `./newtata/masks/`