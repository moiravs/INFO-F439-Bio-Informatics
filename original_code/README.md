# A Multimodal Graph Neural Network Framework for Cancer Molecular Subtype Classification

This is the code for the paper, [A Multimodal Graph Neural Network Framework for Cancer Molecular Subtype Classification](https://arxiv.org/abs/2302.12838).

## Get started
The complete conda enviroment list is in `enviroment.yml` file.

## Demo
For classificaiton on BRCA dataset, use the following command.

```
python cancer_test.py --model gat --num_gene 100 --cancer_subtype True --omic_mode 4 --shuffle_index 0 --gene_gene True --mirna_gene True --mirna_mirna True --parallel True --l2 True --decoder False --poolsize 8 --edge_weight True --epochs 200 --train_ratio 0.7 --test_ratio 0.1
```

## Demo Data
Due the bandwidth limit of Git LFS, we made a mirror host of the demo data on [OneDrive](https://uconn-my.sharepoint.com/:f:/g/personal/bingjun_li_uconn_edu/Euizuh7l_m5DmCq9FX8Vum8BNKBG6k6dpF-RumBfNJrzOA?e=64PG9S).
