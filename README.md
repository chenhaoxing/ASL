# Shaping Visual Representations with Attributes for Few-Shot Learning (IEEE SPL)
This code implements the IEEE Signal Process. Lett. paper Shaping Visual Representations with Attributes for Few-Shot Learning (ASL).

## Citation
If you find our work useful, please consider citing our work using the bibtex:
```
@ARTICLE{asl2022spl,  
        author={Chen, Haoxing and Li, Huaxiong and Li, Yaohui and Chen, Chunlin},  
        journal={IEEE Signal Processing Letters},   
        title={Shaping Visual Representations With Attributes for Few-Shot Recognition},   
        year={2022},  
        volume={29},  
        pages={1397-1401},  
        doi={10.1109/LSP.2022.3180934}
}
```

## Prerequisites
* Linux
* Python 3.7
* Pytorch 1.2
* Torchvision 0.4
* GPU + CUDA CuDNN
## Datasets
You can download datasets automatically by adding `--download` when running the program. However, here we give steps to manually download datasets to prevent problems such as poor network connection:
**CUB**:

1. Create the dir `ASL/datasets/cub`;
2. Download `CUB_200_2011.tgz` from [here](https://drive.google.com/file/d/1hbzc_P1FuxMkcabkgn9ZKinBwW683j45/view), and put the archive into `ASL/datasets/cub`;
3. Running the program with `--download`.

**SUN**:

1. Create the dir `ASL/datasets/sun`;
2. Download the archive of images from [here](http://cs.brown.edu/~gmpatter/Attributes/SUNAttributeDB_Images.tar.gz), and put the archive into `ASL/datasets/sun`;
3. Download the archive of attributes from [here](http://cs.brown.edu/~gmpatter/Attributes/SUNAttributeDB.tar.gz), and put the archive into `ASL/datasets/sun`;
4. Running the program with `--download`.

## Few-shot Classification
Download data and run on multiple GPUs with special settings:

```
python train.py --train-data [train_data] --test-data [test_data] --backbone [backbone] --num-shots [num_shots] --batch-tasks [batch_tasks] --train-tasks [train_tasks] --semantic-type [semantic_type] --multi-gpu --download
```

Run on CUB dataset, ResNet-12 backbone, 1-shot, single GPU

```
python train.py --train-data cub --test-data cub --backbone resnet12 --num-shots 1 --batch-tasks 4 --train-tasks 60000 --semantic-type class_attributes
```

Note that batch tasks are set to 1. Larger batchsize may damage the learning of the visual-attribute generator.

### Our code is based on [TorchMeta](https://github.com/tristandeleu/pytorch-meta).

## Contacts
Please feel free to contact us if you have any problems.

Email: [haoxingchen@smail.nju.edu.cn](haoxingchen@smail.nju.edu.cn)

