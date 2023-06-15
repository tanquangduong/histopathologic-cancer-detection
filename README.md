# Histopathologic cancer detection
:tada:  Build, train, inference custom Resnet50 model using transfer learning for Histopathologic cancer detection


## :white_check_mark: Setup
- Create Python environment\
`conda create -n env_name python=3.10`\
`conda activate env_name`
- Install required packages for production\
`pip install -r .\path_to_requirements\requirements.txt`
- Install required packages for developement\
`pip install -r .\path_to_requirements\requirements-dev.txt`


## :white_check_mark: Train
```
python train.py
```

## :white_check_mark: Inference
```
# Example: inference for single test image
python inference.py --test_image_path ./datasets/cancer_test_dataset/01ca4aacd7904afeba78403d162b3c0fef535944.tif
```