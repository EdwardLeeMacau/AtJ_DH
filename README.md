# AtJ_DH

The folder include all the codes, saved models and testing results prepared for NTIRE 2019 Dehazing Challenge (url: https://competitions.codalab.org competitions/21163)

- Proposed network: "Dense "AtJ" Dehazing Network" 
- Team name: iPAL_AtJ
- Team leader: Tiantong Guo (tong.renly@gmail.com)
- Team members:  Xuelu Li, Venkateswararao Cherukuri, Vishal Monga 
- Time: 03/24/2019

## Requirements
1. Python 3.6
2. py-torch 1.0
3. openCV

## To Run

Navigate to the downloaded folder, in terminal, type:
```
python MAINTHREAD_test.py
```

## Pretrained Model

[here](https://drive.google.com/drive/folders/1d_N-dlj3NJ2tSWQ6wnPJ7tBAmpL-dtm_)

## Post Processing Results

To get the post processing results, please copy please go to [here](https://drive.google.com/drive/folders/1Ezzwa-dg15UbkPYZIAd3BNb7Zi5cJ-35?usp=sharing)

(the results generated from this code is pre-loaded in that post-processing folder)
(user can also copy the results from result_AtJ_model_GPU to the post-processing ‘testsets’ folder)

## Files

1. MAINTHREAD_test.py:

	The file includes test codes for our proposed network.

	```
	Parser arguments: 
		--cuda (default = True) for running on GPU to reproduce the submitted results, 
		--model (default = "AtJ_model") for the model path, 
		--test (default="testset") for the testset path.
	```
	Return: running time and the saving path for results.

2. Model:

	The folder includes the network model structure and the saved pretrained model.

   1. `__init__.py`: required for Python to recognize the package location.
   2. `AtJ_model.pth`: the pretrained model for our proposed network using images provided in NTIRE 19 Dehazing Challenge (and the synthetic dense haze images using the data from NTIRE 18)
   3. `AtJ_model.py`: the network structure for our "AtJ" model

3. result_AtJ_model_GPU: 

	The folder includes the dehazed images of the test images provided in NTIRE 19 Dehazing Challenge.

	1. `*.png`: the dehazed images using our proposed network.
	2. `readme.txt`: the information for running time, GPU and Data usage

4. testset:

	The folder includes the test images from NTIRE 19 Dehazing Challenge.

5. utils:

	The folder includes small snippets that can be used though the application.

	1. `utils.py`: function codes that can be used through the application.
