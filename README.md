# pytorch1.0_MostUse
> Some mostly-used code and some mostly-happened problems in PyTorch 1.0

## Use Google Colab
  
**Chinese Students must use vpn !!!**
  
**Using Google GPU, free but of good performance.**
  
1. GPU Profile  
    <p>
      <img src="https://github.com/lcylmhlcy/pytorch1.0_MostUse/raw/master/img/1.png" width=600>
    </p>
    
    ```
    # Mount Google Cloud Disk
    from google.colab import drive
    drive.mount('/content/drive')
    ```
    
    <p>
      <img src="https://github.com/lcylmhlcy/pytorch1.0_MostUse/raw/master/img/10.png" width=600>
    </p>
    
    <p>
      <img src="https://github.com/lcylmhlcy/pytorch1.0_MostUse/raw/master/img/11.png" width=200>
      <img src="https://github.com/lcylmhlcy/pytorch1.0_MostUse/raw/master/img/12.png" width=200>
    </p>

2. PyTorch Profile  
    <p>
      <img src="https://github.com/lcylmhlcy/pytorch1.0_MostUse/raw/master/img/2.png" width=600>
    </p>

3. [Some mostly-used code](https://github.com/lcylmhlcy/pytorch1.0_MostUse/blob/master/pytorch1_0_.ipynb)

4. [some mostly-happened problems](https://github.com/lcylmhlcy/pytorch1.0_MostUse/blob/master/some_problems.md)  

5. [ignite: High-level library to help with training neural networks in PyTorch](https://github.com/lcylmhlcy/pytorch1.0_MostUse/tree/master/ignite)  
    - [TensorboardX](https://github.com/lanpa/tensorboardX): tensorboard for pytorch
    - [Visdom](https://github.com/facebookresearch/visdom): A flexible tool for creating, organizing, and sharing visualizations of live, rich data. 
    - PyTorch 1.1 has included tensorboard. **[torch.utils.tensorboard](https://pytorch.org/docs/stable/tensorboard.html)  

6. [torchvision functions](https://github.com/lcylmhlcy/pytorch1.0_MostUse/blob/master/torchvision.md)  
    - [torchvision 0.3 Blog](https://github.com/pytorch/vision/releases)

7. [TVM](https://github.com/dmlc/tvm): End to End Deep Learning Compiler Stack  
    - TVM integration into PyTorch: https://github.com/pytorch/tvm  

8. [Glow](https://github.com/pytorch/glow): Compiler for Neural Network hardware accelerators  

9. [Ax](https://github.com/facebook/Ax) + [BoTorch](https://github.com/pytorch/botorch): Ax is an accessible, general-purpose platform for understanding, managing, deploying, and automating adaptive experiments. **Bayesian optimization in Ax is powered by BoTorch, a modern library for Bayesian optimization research built on PyTorch.**  

10. [flashtorch](https://github.com/MisaOgura/flashtorch): Visualization toolkit for neural networks in PyTorch - towards explainable and interpretable AI!
