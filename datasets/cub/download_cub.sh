wget "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz"
tar -zxvf CUB_200_2011.tgz

wget "https://raw.githubusercontent.com/OML-Team/open-metric-learning/main/pipelines/datasets_converters/convert_cub.py"
python convert_cub.py --dataset_root=CUB_200_2011 --no_bboxes
