# Cross-modal Image-text Retrieval Based on Multi-modal Mask and Hash Constraint

## Usage
### Requirements
we use single RTX3080 8G GPU for training and evaluation. 
```
For the required environment, see environment.yaml
```

### Prepare Datasets
Download the CUHK-PEDES,CUB-200,flickr30k,ICFG-PEDES

To change the dataset, just modify the config file, and then read the different config file from the command line


## Training

```python
train.sh
```

## Testing

```python
python test_net.py 
```

## Results
![image-20230612160035367](C:\Users\10211\AppData\Roaming\Typora\typora-user-images\image-20230612160035367.png)

#### ICFG-PEDES dataset

![image-20230612160113537](C:\Users\10211\AppData\Roaming\Typora\typora-user-images\image-20230612160113537.png)

#### CUHK-PEDES dataset

![image-20230612160058984](C:\Users\10211\AppData\Roaming\Typora\typora-user-images\image-20230612160058984.png)



