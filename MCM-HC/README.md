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
python test.py --config-file XXpath --checkpoint-file XXpath
```

## Results
![image](https://github.com/CCNU-DigitalLibrary/CCNU-DigitalLibrary/assets/135103900/c78f3e91-38ad-4b69-8c04-2f810595b45a)

#### ICFG-PEDES dataset

![image](https://github.com/CCNU-DigitalLibrary/CCNU-DigitalLibrary/assets/135103900/8dbdacb5-e9ed-4c58-8e82-a43d7349e3d1)

#### CUHK-PEDES dataset

![image](https://github.com/CCNU-DigitalLibrary/CCNU-DigitalLibrary/assets/135103900/4baf5353-61ab-4cd5-85cf-a1e40a1be433)


