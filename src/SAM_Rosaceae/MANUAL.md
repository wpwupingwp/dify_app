# Rosaceae SAM

## crawler.py

Collect 14 (`species_list.csv`) Rosaceae species' images from PPBC.
Split to train set and test set.

## Preprocess
1. detect and remove duplicate images

Calculate hash and compare.

   - Normal hash: strict identical
   - Manual: slow, for verification
   - pHASH: allow small difference 
      - https://github.com/idealo/imagededup/tree/master
      - https://github.com/xuehuachunsheng/DupImageDetection
2. detect and remove low quality images
    CLIP-iqa method, with score.py
3. cluster images
   - https://github.com/LexCybermac/smlr
   - https://blog.bruun.dev/semantic-image-clustering-with-clip/
4. organize dataset. a. train/test; b. species. c. whole plant/organ. d. organ
5. SAM label. Manual or automatic?
    - https://cloud.tencent.com/developer/article/2315237
    - https://github.com/zhouayi/SAM-Tool
    - https://docs.ultralytics.com/zh/tasks/segment/
6. finetune
    - SAM or EfficientSAM?
    - SAM only
    - YOLO then SAM
    - https://github.com/yformer/EfficientSAM
    - https://github.com/facebookresearch/segment-anything
    - https://blog.51cto.com/u_15699099/6237305
    - https://blog.csdn.net/jcfszxc/article/details/136181686
    - https://zhuanlan.zhihu.com/p/622677489
    -
7. test
