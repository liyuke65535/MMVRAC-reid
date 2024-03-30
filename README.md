## [2024 ICME Grand Challenge: Multi-Modal Video Reasoning and Analyzing Competition (MMVRAC)](https://sutdcv.github.io/MMVRAC/)

dataset: [UAVhuman-reid](https://github.com/sutdcv/UAV-Human)

### 1. Configurations
First of all, create a conda env, then install packages detailed in enviroments.sh
```
conda create -n reid python==3.9
conda activate reid
bash enviroments.sh
```

Note that, all experiments are conducted using single GPU: NVIDIA Titan RTX.

### 2. training
We use the model pretrained on Market+DukeMTMC+MSMT17+cuhk02+cuhk03+CUHK-SYSU.

Modify the paths and settings in config/pretrained_vit.yml, then

```
python train.py --config_file config/#your_config_name#.yml
```

### 3. evaluation
We use multiple tricks to re-rank the ranking lists of queries, like re-ranking, rank fusion of multiple models. Besides, we use the setting of multi-shot query images to further improve the performance.

Modify TEST.WEIGHT as your trained model path in config/uavhuman.yml, then

```
python test.py --config_file config/uavhuman.yml
```

Otherwise, specifying the trained path in terminal is also valid:

```
python test.py --config_file config/uavhuman.yml TEST.WEIGHT ${your trained model path}
```

### 4. evaluate with re-ranking
Re-ranking is provided for better retrieval results as well.

Reference: [Re-ranking Person Re-identification with K-reciprocal Encoding](https://openaccess.thecvf.com/content_cvpr_2017/papers/Zhong_Re-Ranking_Person_Re-Identification_CVPR_2017_paper.pdf).

The default setting of UAVHuman: k1=4, k2=4, lambda=0.45.
```
python test_with_reranking.py --config_file config/uavhuman.yml TEST.WEIGHT ${your trained model path}
```
