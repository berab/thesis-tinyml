# Multi-objective optimization of Audio Processing DNNs on Embedded Hardware

## Description:
As part of this master thesis, the following investigations need to be carried out. 
- Get familiar with the state-of-the-art models for the two audio process- ing applications. 
- Utilize existing optimization techniques with the audio processing ap- plications. 
- Select the methods that are deployable on the available embedded hardware. 
- Develop a hardware-aware multi-objective optimization flow. 
- Target a specific objective for innovation or competition with the state- of-the-art. 

## How to get started with the baseline
First you have to setup the required environment (Download dcase20 dataset):
```bash
python -m venv envs/torch
source envs/torch/bin/activate
pip install -r requirements.txt
./dcase.sh
export CUBLAS_WORKSPACE_CONFIG=:16:8
```

After that everything should be setup to run tests:
```bash
pytest src/*
```

Now you can run the baseline for several audio processing application: 
```bash
python src/main.py
```

Set --config-name=asc3 or sc for different applications. Take a look inside YAML files to see possible configurations. 

## Pruning
You can run lottery ticket experiments or training with iterative pruning: 
```bash
python src/main.py apply=lottery
python src/main.py apply=itr-prune
```

Or you can just prune a pretrained model: 
```bash
python src/main.py apply=prune model_dir=path/to/model/state
```

## Quantization
Setup the required environment for NEMO: 
```bash
python -m venv envs/nemo
source envs/nemo/bin/activate
pip install -r requirements-nemo.txt
```

You can post-training quantization and fine-tuning with NEMO or Nvidia: 
```bash
python src/main.py model.quant=nemo model_dir=path/to/model/state optim=sgd-steplr optim.optim.lr=0.001 n_epochs=5
python src/main.py model.quant=nvidia model_dir=path/to/model/state optim=sgd-steplr optim.optim.lr=0.001 n_epochs=5
```

Or you can run quantization-aware-training with Brevitas(Currently only for SC): 
```bash
python src/main.py --config-name=sc model.quant=brevitas
```

## Knowledge distillation
```bash
python src/main.py --config-name=sc tmodel=vgg19 tmodel_dir=/path/to/teacher/state model_dir=/path/to/student/state optim.optim.lr=0.001
```

## Testing
You can test a pre-trained network: 
```bash
python src/main.py apply=test model_dir=path/to/model/state
```

## Resources
-	ASC model: [QTI SUBMISSION TO DCASE 2021: RESIDUAL NORMALIZATION FOR DEVICE-IMBALANCED ACOUSTIC SCENE CLASSIFICATION WITH EFFICIENT DESIGN](https://arxiv.org/abs/2111.06531) 
-	SC model: [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
-	[DCASE2022](https://dcase.community/challenge2022/)
-	[Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition](https://arxiv.org/abs/1804.03209)
-	[NEMO (NEural Minimizer for pytOrch)](https://github.com/pulp-platform/nemo) 
-	[Brevitas](https://github.com/Xilinx/brevitas)

## Compression techniques
-	[The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks](https://arxiv.org/abs/1803.03635)
-	[Extremely Low Bit Neural Network: Squeeze the Last Bit Out with ADMM](https://arxiv.org/abs/1707.09870)

