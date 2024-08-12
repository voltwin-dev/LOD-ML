# LOD: Learnable Orthogonal Decomposition
![](https://i.imgur.com/waxVImv.png)
![KO-platypus](./images/LOD.png)

# Table of Contents
1. [Introduction]()
2. [Datasets]()
3. [POD preprocess]()
4. [LOD Training]()
5. [Inference]()
6. [LOD Visualization]()
7. [References]()

# Introduction
Understanding spatio-temporal data is a central challenge in the field of deep learning, particularly in solving Partial Differential Equations (PDEs). Existing approaches, such as Transformers and Neural Operators, have successfully mapped input conditions to PDE solutions but often face challenges due to their auto-regressive nature, which leads to increased computational costs and error accumulation over time. In this paper, we introduce a novel approach called Learnable Orthogonal Decomposition (LOD), inspired by the classical Proper Orthogonal Decomposition (POD) technique and enhanced by deep learning. LOD effectively decouples temporal and spatial information in PDE data, simplifying the learning process for neural networks. By focusing on the core architecture of LOD, the method is designed to maximize efficiency while maintaining high accuracy. Spatial bases are initialized with POD-generated components, which are set as learnable parameters within the model. The deep learning model then predicts the temporal coefficients in a single inference step, enabling non-regressive prediction of the entire time series. Experiments show that LOD not only accurately captures the dynamics of complex physical systems but also outperforms traditional POD methods, achieving high accuracy and low computational cost across various PDE benchmark datasets.
  
# Datasets
Before training, you must download the [PDEBench dataset](https://github.com/pdebench/PDEBench/tree/main/pdebench/data_download).  
You can download above link, also can see [our code]().

Here, below is folder structure we recommend.
```
data2
├── PDEBench                  
│   ├── 1D
│       ├── Advection
│       ├── Burgers
│       ├── CFD
│       ├── ReactionDiffusion
│       ├── diffusion-sorption
│   ├── 2D           
│       ├── shallow-water
```
  
# POD preprocess
## 1D-PDE & CFD
Use the [make_1D_pod]() yaml files.
```yaml
dataset:
    root_path: '/data2/PDEBench/1D'
    save_path: '/data2/PDEBench/POD/' # We recommend
    data_path: ['1D_diff-sorp_NA_NA.h5'] # PDE data
    N_eigen: 64 # POD Hyperparameters
```
  
Then, implement below clode.
```python
python POD_1D_process.py
```
  
If you want to preprocess about CFD,
```python
python POD_1D_process.py --pde CFD
```
  
## Shallow-water

# LOD Training
## 1D-PDE
- Advection, Burgers, Diffusion-Reaction, and Diffusion-Sorption
```python
python LOD_1D.py --pde [choose ...advection, burgers, reaction, sorption...]
```
  
- 1D-CFD
```python
python LOD_CFD.py
```  
  
## Shallow-water
```python
python LOD_2D.py
```
  
# Inference
(TODO)

# LOD Visualization
(TODO)

# References
- [PDEBench](https://github.com/pdebench/PDEBench)
- [VCNeF](https://github.com/jhagnberger/vcnef)
