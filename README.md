# Intel(R) Extension for PyTorch*(IPEX) and PyTorch Distributed Data Parallel(DDP) with Intel(R) oneCCL demo 

## Intel(R) Extension for PyTorch*(IPEX) demo 
You need to git clone the repo using :
```
git clone https://github.com/oneapi-src/oneAPI-samples.git
```
After having the repo locally, you should see "oneAPI-samples" directory. Navigate to **"oneAPI-samples/AI-and-Analytics/Features-and-Functionality/"**


 For IntelR) Extension for PyTorch lab sample, open **IntelPyTorch_Extensions_Inference_Optimization/optimize_pytorch_models_with_ipex.ipynb** <br>
 Select the kernel "PyTorch" if not auto-selected. Now you are ready to run all the cells one by one.

 
## PyTorch DDP with oneCCL sample
1. After downloading the repo. Change the file permission for execution using: 
```
chmod 755 run.sh
```

2. Execute the run.sh script using: 
```
./run.sh
```
Please Note: First run downloads the dataset if there is some error related to dataset, please run it again      




## Intel(R) Extension for Scikit-learn* sample(Additional):

 For Intel(R) Extension for Scikit-learn lab sample, open Intel_Extension_For_SKLearn_Performance_SVC_Adult/Intel_Extension_for_SKLearn_Performance_SVC_Adult.ipynb
 Select the kernal "Python 3 (Intel® oneAPI)" if not auto-selected. Now you are ready to run all the cells one by one.
 

## Create PyTorch+IPEX+Transformers env. In case your PyTorch env/kernal is throwing error while importing IPEX(Optional).

1. In the DevCloud terminal, Git clone this repo. 
```
"git clone https://github.com/asirvaiy/H2S_oneAPI.git"
```
2. Go to the repo. 
```
cd H2S_oneAPI
```
3. Execute the pip_env.sh file for creating the environment env_h2s1api and kernal named "IPEX" for PyTorch+IPEX+Transformers. 
```
bash pip_env.sh
```

