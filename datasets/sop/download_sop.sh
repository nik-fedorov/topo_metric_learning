python -c "import gdown; gdown.download('https://drive.google.com/uc?id=1TclrpQOF_ullUP99wk_gjGN8pKvtErG8')"
unzip Stanford_Online_Products.zip

wget "https://raw.githubusercontent.com/OML-Team/open-metric-learning/main/pipelines/datasets_converters/convert_sop.py"
python convert_sop.py --dataset_root=Stanford_Online_Products
