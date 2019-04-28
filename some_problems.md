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
- Data is not on the same GPU while using nn.Dataparallel.  
  **Put the data on the same GPU by using `.cuda()`.**  
  
- nn.Dataparallel model load  
  **Use model.module instead.**
  ```
  def get_model(self):
    if self.nGPU == 1:         
        return self.model     
    else:         
        return self.model.module 
  ```

-  
