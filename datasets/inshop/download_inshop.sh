mkdir -p DeepFashion_InShop
pushd DeepFashion_InShop

## download from my gdrive
python -c "\
import gdown; \
gdown.download('https://drive.google.com/uc?id=1qx69Q2QmQ451tt_g1tDR5W1SnAj_dpfg'); \
gdown.download('https://drive.google.com/uc?id=1rsyRGOOUAdcXVxKdcsybr0qM4VZERlVj'); \
gdown.download('https://drive.google.com/uc?id=1bzmNYYj3b4A_Llp9kio1GIxhFqTxrUCN'); \
"

## official gdrive links:
#gdown.download('https://drive.google.com/uc?id=1bByKH1ciLXY70Bp8le_AVnjk-Hd4pe_i'); \
#gdown.download('https://drive.google.com/uc?id=0B7EVK8r0v71pYVBqLXpRVjhHeWM'); \
#gdown.download('https://drive.google.com/uc?id=0B7EVK8r0v71pMGpUY0x2aEtvcFE'); \
unzip img_highres.zip
popd

wget "https://raw.githubusercontent.com/OML-Team/open-metric-learning/main/pipelines/datasets_converters/convert_inshop.py"
python convert_inshop.py --dataset_root=DeepFashion_InShop --no_bboxes
