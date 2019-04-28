#### 1. nn.Module.cuda() and Tensor.cuda() have different effects.  
```
nn.Module:
model = model.cuda() # migrate the memory of model itself
model.cuda() # the same with above

nn.Tensor:
tensor = tensor.cuda() # Reassign a new GPU tensor of the tensor
tensor.cuda() # Returns a copy of the tensor in GPU memory without changing itself.
```
  
#### 2. torch.Tensor.detach()  
Offical: Returns a new Tensor, detached from the current graph. The result will never require gradient.  
```
# If Assuming there are model A and model B, we need to take the output of A as the input of B, but we only train model B when we train. Then we can do this:

input_B = output_A.detach()

# It can disconnect the gradient transfer between the two computational graphs, thus realizing the functions we need.
```
  
#### 3. CrossEntropyLoss  
Offical: CrossEntropyLoss(self, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='elementwise_mean')  
- If reduce = False, size_average is no-use. return the loss vector, that is losses for each element in batch.
- If reduce = True，return the scalar loss:
  - If size_average = True，return loss.mean().
  - If size_average = False，return loss.sum().
- weight: Enter a 1D weighting vector, weighting loss for each category.
- ignore_index: Select the target value to be ignored so that it does not contribute to the input gradient. If size_average = True, then only the loss mean of the non-ignored target is calculated.
- reduction : 'none' | 'elementwise_mean' | 'sum'

#### 4. Warm up: possible improvements in model accuracy  
```
if ep < 50:
   lr = 1e-4*(ep//5+1) # very low learning rate
 elif ep < 200:
   lr = 1e-3
 elif ep < 300:
    lr = 1e-4
```
#### 5. nn.Dataparallel
Offical: torch.nn.DataParallel(model, deviceids, outputdevice, dim)
```
model = nn.DataParallel(model,device_ids=[1,3])
model = model.cuda(device_ids[0]) # default is GPU 0. If this, all tensors must be on GPU device_ids[0]
```
  
- Data is not on the same GPU while using nn.Dataparallel.  
  **Put the data on the same GPU by using `.cuda()`. All tensors must be on the same SPU.**  
  
- nn.Dataparallel model load  
  **Use model.module instead.**  
  ``` 
  # Usual
  model = Model()
  model.fc

  # DataParallel
  parallel_model = torch.nn.DataParallel(model)
  parallel_model.fc  # Wrong
  parall_model.module.fc # Right
  ```

#### 6. Pretrained-models input  
All pre-trained models expect input images normalized in the same way, i.e. mini-batches of 3-channel RGB images of shape (3 x H x W), where H and W are expected to be at least 224. The images have to be loaded in to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225]


#### 7. RuntimeError: Trying to backward through the graph a second time, but the buffers have already been freed. Specify retain_graph=True when calling backward the first time.  
The problem is that by default, the network does not allow multiple backwards() in back propagation. You need to set retain_graph = True in the first backward.
```
loss.backward(retain_graph=True)
```

