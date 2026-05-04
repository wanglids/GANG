# GANG: Geometrically-Aligned Neural Gaussians for Efficient and Realistic Relighting [IEEE TVCG 2026]


### [Paper Link](https://ieeexplore.ieee.org/abstract/document/11498572/)

***
Deqi Li<sup>1,2</sup>, Shi-Sheng Huang<sup>1</sup>, Hongbo Fu<sup>3</sup>, Hua Huang<sup>1✉</sup>

<sup>1</sup>Beijing Normal University; <sup>2</sup>Tsinghua University;<sup>3</sup>Hong Kong
University of Science and Technology; <sup>✉</sup>Corresponding Author.
***

![block](assets/teaser.pdf)   
![block](assets/pipeline.pdf)   



## Environmental Setups
```bash
git clone https://github.com/wanglids/GANG
cd GANG
conda create -n GANG python=3.8
conda activate GANG

pip install -r requirements.txt
pip install -e submodules/light_gaussian
pip install -e submodules/nvdiffrast-main
pip install -e submodules/permutohedral_encoding-main
pip install -e submodules/simple-knn
```
In our environment, we use `pytorch=2.0.1+cu118`. For torch_scatter, we recommend using the `.whl` file for a faster installation. First, navigate to the submodules directory and then install using the appropriate `.whl` file:
```
cd submodules
pip install torch_scatter-2.1.2+pt20cu118-cp38-cp38-linux_x86_64.whl
```
Alternatively, you can download the `.whl` file from the PyTorch Geometric official website. Visit the following [link](https://pytorch-geometric.com/whl/), choose the version corresponding to your environment, and install it.

## Data Preparation
We conducted experiments on the datasets [`Mip-NeRF 360`](https://jonbarron.info/mipnerf360/), [`Deep T&T`](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip) and [`TensoIR Synthetic`](https://zenodo.org/records/7880113#.ZE68FHZBz18). You can organize your own data in the same way as the original data structures of [3DGS](https://github.com/graphdeco-inria/gaussian-splatting) and [TensoIR](https://github.com/Haian-Jin/TensoIR). The environment map is from TensoIR and can be downloaded from [here](https://drive.google.com/file/d/10WLc4zk2idf4xGb6nPL43OXTTHvAXSR3/view).

## Training
For training, some scripts are located in the `\script` directory. You can modify them according to your own data. During quick tests, you can execute the following command: the "train.py" file contains the test rendering.

``` 
bash train.sh
``` 
You can customize your training parameters via configuration files. 
## Rendering
Run the following script to render the images.  
```
bash render.sh
```


## Scripts

There are some helpful scripts in `scripts/`, please feel free to use them.

---

## Citation
If you find this code useful for your research, welcome to cite the following paper:
```
@ARTICLE{11498572,
  author={Li, Deqi and Huang, Shi-Sheng and Fu, Hongbo and Huang, Hua},
  journal={IEEE Transactions on Visualization and Computer Graphics}, 
  title={GANG: Geometrically-Aligned Neural Gaussians for Efficient and Realistic Relighting}, 
  year={2026},
  volume={},
  number={},
  pages={1-15},
  doi={10.1109/TVCG.2026.3687668}}
```
## Acknowledgments

Some source code of ours is borrowed from [Octree-GS](https://github.com/city-super/Octree-GS),[GS-IR](https://github.com/lzhnb/GS-IR) and [PhySG](https://github.com/XPandora/PhysGaussian). We sincerely appreciate the excellent works of these authors.
