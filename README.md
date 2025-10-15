# DRSN
Official implementation of "Dynamic Recurrent Self-refinement Network for Hyperspectral Remote Sensing Image Super-Resolution"

## Dependencies and Installation
```
git clone https://github.com/hgpftd/DRSN.git
cd DRSN

conda create -n drsn python=3.9 -y
conda activate drsn

pip install -r requirements.txt
```


## Preparations of dataset
We employ three benchmark hyperspectral image datasets in our experiments: Houston, Chikusei, and Pavia Centre. All datasets can be downloaded from their official websites listed below:

1,[Houston](https://naotoyokoya.com/Download.html)    
2,[Chikusei](https://naotoyokoya.com/Download.html)   
3,[Pavia centre](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes)  


## Usage
### Testing
You can use the following command for testing, e.g., for 8× SR on the Houston dataset:
```
python test.py --cuda --datasetName Houston --n_bands 48 --upscale_factor 8 --model_name ./weight/Houston_8_epoch_300.pth
```

### Training
You can use the following command for training, e.g., for 8× SR on the Houston dataset:
```
python -m torch.distributed.launch --nproc_per_node=1 --master_port=22024 --use_env train.py --cuda --datasetName Houston --n_bands 48 --upscale_factor 8
```
You can also use the framework of other HSISR methods, e.g., [MCNet](https://github.com/qianngli/MCNet) and [SSPSR](https://github.com/junjun-jiang/SSPSR).

## Results
### Visual comparisons 
<img src="assets/Visualization_Houston.jpg" width="800px"/>

<img src="assets/Visualization_Pavia.jpg" width="800px"/>

### Quantitative comparison

<img src="assets/Quantitative_Houston.jpg" width="800px"/>


## Acknowledgement
We thank everyone who makes their code and models available, especially [MCNet](https://github.com/qianngli/MCNet), [GELIN](https://github.com/HuQ1an/GELIN_TGRS) and [RFSR](https://github.com/wxywhu/RFSR_TGRS). Thanks for their awesome works.

