# pytorch_MostUse
> Some mostly-used code and some mostly-happened problems in PyTorch

## Use Google Colab 
- GPU Profile  
    <p>
      <img src="https://github.com/lcylmhlcy/pytorch_MostUse/raw/master/img/1.png" width=600>
    </p>   
    ```
    # Mount Google Cloud Disk
    from google.colab import drive
    drive.mount('/content/drive')
    ``` 
    <p>
      <img src="https://github.com/lcylmhlcy/pytorch_MostUse/raw/master/img/10.png" width=600>
    </p>  
    <p>
      <img src="https://github.com/lcylmhlcy/pytorch_MostUse/raw/master/img/11.png" width=200>
      <img src="https://github.com/lcylmhlcy/pytorch_MostUse/raw/master/img/12.png" width=200>
    </p>
- PyTorch Profile  
    <p>
      <img src="https://github.com/lcylmhlcy/pytorch_MostUse/raw/master/img/2.png" width=600>
    </p>


## Notebook
- [Some mostly-used code](https://github.com/lcylmhlcy/pytorch_MostUse/blob/master/pytorch1_0_.ipynb)
- [some mostly-happened problems](https://github.com/lcylmhlcy/pytorch_MostUse/blob/master/some_problems.md)
- [Pytorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) [[Documentation](https://pytorch-lightning.readthedocs.io/en/stable/)]
- [wandb](https://wandb.ai/site) [[Doc](https://docs.wandb.ai/)]
  - [**pytorch lightning + wandb**](https://pytorch-lightning.readthedocs.io/en/stable/extensions/generated/pytorch_lightning.loggers.WandbLogger.html#pytorch_lightning.loggers.WandbLogger)

## Others
- [torchvision functions](https://github.com/lcylmhlcy/pytorch_MostUse/blob/master/torchvision.md)
- [TVM](https://github.com/dmlc/tvm): End to End Deep Learning Compiler Stack  
    - TVM integration into PyTorch: https://github.com/pytorch/tvm  
- [Glow](https://github.com/pytorch/glow): Compiler for Neural Network hardware accelerators  
- [Ax](https://github.com/facebook/Ax) + [BoTorch](https://github.com/pytorch/botorch)
  - Ax is an accessible, general-purpose platform for understanding, managing, deploying, and automating adaptive experiments. **Bayesian optimization in Ax is powered by BoTorch, a modern library for Bayesian optimization research built on PyTorch.**  
- [flashtorch](https://github.com/MisaOgura/flashtorch)-
  - Visualization toolkit for neural networks in PyTorch - towards explainable and interpretable AI!
