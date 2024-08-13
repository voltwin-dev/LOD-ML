# LOD: Learnable Orthogonal Decomposition🔥
![](https://i.imgur.com/waxVImv.png)
![KO-platypus](./images/LOD.png)

# Table of Contents📖
1. [Introduction]()
2. [Datasets]()
3. [POD preprocess]()
4. [LOD Training]()
5. [Inference]()
6. [LOD Visualization]()
7. [References]()

# Introduction📖
Understanding spatio-temporal data is a central challenge in the field of deep learning, particularly in solving Partial Differential Equations (PDEs). Existing approaches, such as Transformers and Neural Operators, have successfully mapped input conditions to PDE solutions but often face challenges due to their auto-regressive nature, which leads to increased computational costs and error accumulation over time. In this paper, we introduce a novel approach called Learnable Orthogonal Decomposition (LOD), inspired by the classical Proper Orthogonal Decomposition (POD) technique and enhanced by deep learning. LOD effectively decouples temporal and spatial information in PDE data, simplifying the learning process for neural networks. By focusing on the core architecture of LOD, the method is designed to maximize efficiency while maintaining high accuracy. Spatial bases are initialized with POD-generated components, which are set as learnable parameters within the model. The deep learning model then predicts the temporal coefficients in a single inference step, enabling non-regressive prediction of the entire time series. Experiments show that LOD not only accurately captures the dynamics of complex physical systems but also outperforms traditional POD methods, achieving high accuracy and low computational cost across various PDE benchmark datasets.
  
# Datasets📚
Before training, you must download the [PDEBench dataset](https://github.com/pdebench/PDEBench/tree/main/pdebench/data_download).  
You can download above link, also can see [our code](https://github.com/voltwin-dev/LOD-ML/blob/main/dataset/download_pdebench.ipynb).

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
  
# POD preprocess🌊
## 1D-PDE & CFD
Use the [make_1D_POD](https://github.com/voltwin-dev/LOD-ML/blob/main/config/make_1D_POD.yaml) yaml files.
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
python POD_1D_CFD_process.py
```
  
## Shallow-water
(TODO)

# LOD Training🤗
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
  
# Evaluation🌟
## Efficiency
<img src="./images/vram.png" width="400" height="400" />
| Model | Inference_Time | VRAM | # of parameters |
| --- | --- | --- | --- | 
| **FNO** | 43s | 34.83MB | 43137 | 
| **LOD-small** | **3s** | 34.83MB | 43325 |
| **LOD** | **7s** | 51.96MB | 4451194 |
  
LOD-small is about **14.3x** faster than FNO.  
LOD have a higher VRAM usage than FNO, but about **6.1x** faster inference speed.
  
## Performance
(TODO)
  
# LOD Inference🌊
## Code
```python
python 1D_visualization.py --pde [choose ...advection, burgers, reaction, sorption...]
```
We provided some [checkpoints](https://drive.google.com/drive/folders/1yDLMFUdq16HwNyIJfMvCFoxqdmnKfyQ1?usp=sharing).  
You will easily implement our code..!  
```
checkpoint
├── lod_Advection_beta0.1.pt
├── lod_Burgers_Nu1.0.pt
├── lod_ReactionDiffusion_Nu0.5_Rho1.0.pt
└── lod_ReactionDiffusion_Nu5.0_Rho10.0.pt
```
  
## Results
<table class="center">
<tr>
  <td style="text-align:center;" colspan="3"><b>LOD results</b></td>
</tr>
<tr>
  <td><img src="./images/adv_LOD_705.gif"></td>
  <td><img src="./images/adv_LOD_777.gif"></td>
  <td><img src="./images/adv_LOD_560.gif"></td>              
</tr>
<tr>
  <td width=25% style="text-align:center;">"Advection case1”</td>
  <td width=25% style="text-align:center;">"Advection case2"</td>
  <td width=25% style="text-align:center;">"Advection case3"</td>
</tr>

<tr>
  <td><img src="./images/bur_LOD_860.gif"></td>
  <td><img src="./images/bur_LOD_950.gif"></td>
  <td><img src="./images/bur_LOD_235.gif"></td>              
</tr>
<tr>
  <td width=25% style="text-align:center;">"Burgers case1”</td>
  <td width=25% style="text-align:center;">"Burgers case2"</td>
  <td width=25% style="text-align:center;">"Burgers case3"</td>
</tr>

<tr>
  <td><img src="./images/react_LOD_999.gif"></td>
  <td><img src="./images/react_LOD_0.gif"></td>
  <td><img src=""></td>              
</tr>
<tr>
  <td width=25% style="text-align:center;">"Diffusion-Reaction case1”</td>
  <td width=25% style="text-align:center;">"Diffusion-Reaction case2"</td>
  <td width=25% style="text-align:center;">"Shallow-Water"</td>
</tr>

</table>

# References
- [PDEBench](https://github.com/pdebench/PDEBench)
- [VCNeF](https://github.com/jhagnberger/vcnef)
