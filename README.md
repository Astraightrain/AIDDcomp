# dacon Graph Transformer(GPS) multitask model inference module

## Environment setting & installation
```bash
conda env create -f requirements.yml
conda activate dacon_gps
conda install -c conda-forge torch-scatter
conda install pyg -c pyg
pip install -e .
```
## Download weight files
```
Link: contact -> sanice1229@gmail.com
saved_model folder 전체 다운로드 후 dacon repo directory로 옮기기
dacon   -gps
        -results
        -saved_model <-
        -inference.py
        -README.md
        -setup.py
        ...


```

# Basic usage
```
python inference.py --dataset ./dataset/test_data.csv --output output_name.csv
# inference output will be saved at results directory.
```
