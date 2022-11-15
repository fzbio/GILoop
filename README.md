# GILoop
GILoop is a deep learning model for detecting CTCF-mediated loops on Hi-C contact maps. 

![Model architecture](./figure/architecture.png)



**New:** *GILoop now supports `.cool` input*



#### Installation:

```
conda create -n GIL python=3.8
conda activate GIL
pip install -r requirements.txt
```

After running the code segment above, please download `models` and `data` from [GILoop_assets](https://portland-my.sharepoint.com/:f:/g/personal/fuzhowang2-c_my_cityu_edu_hk/EpsC_y58ARNInLGjwy4yc44BNs2fKzCXNFVLUxrsrtHO2A?e=83KzE4) and **replace the ones in the local directory**. 



#### Usage:

You can either use `.cool` files as input, or use the output format of Juicer dump as input.

**Cooler input:**

```
python demo.py
```

API usage is self-documented in [demo.py](./demo.py). This script is the full pipeline including data pre-processing (sequencing depth alignment, normalization, and expected vector calculation, etc.), and GILoop sampling and training. The preprocessing overhead could be heavy and could lead to a large running time. 

**Juicer dump text file as input:**

```
python demo_from_processed.py
```

In this case, please make sure the text files are KR-normalized O/E matrices, and the source and the target are of similar number of reads. The script only carries out GILoop patching and training. No preprocessing is applied to the data. 