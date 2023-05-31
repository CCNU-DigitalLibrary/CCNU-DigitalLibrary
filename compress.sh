#!/bin/bash
tar --exclude='./output' --exclude='./pretrained'  --exclude='._.DS_Store' --exclude='./datasets/cuhkpedes' --exclude='./datasets/icfgpedes' --exclude='./datasets/flickr' -zvcf HashTextReID-v3.tar.gz .

