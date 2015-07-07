-- linear regression
-- https://github.com/torch/demos/blob/master/linear-regression/example-linear-regression.lua
-- run with:
--  torch practical2.lua
--  torch -i practical2.lua

require 'torch'
require 'optim'
require 'nn'

-- Each row represents a training sample and each column a variable
-- First column is the target variable and the others are input

data = torch.Tensor{
  {40,  6,  4},
  {44, 10,  4},
  {46, 12,  5},
  {48, 14,  7},
  {52, 16,  9},
  {58, 18, 12},
  {60, 22, 14},
  {68, 24, 20},
  {74, 26, 21},
  {80, 32, 24}
}

-- 2 input for input layer, 1 output for output layer
-- output of each module become the inputs of the subsequent module

model = nn.Sequential() -- container
ninputs = 2; noutputs = 1
model:add(nn.Linear(ninputs, noutputs))

criterion = nn.MSECriterion() -- minimize Mean Square Error (MSE)

-- Stochastic Gradient Descent (SGD)

-- x : trainable parameters of model
-- dl_dx : gradient of loss function with regard to parameters x
x, dl_dx = model:getParameters()



-- feval : compute the value of the loss function at a given point x and the gradient of loss function with respect to x
-- x : here x is the vector of trainable weights (all the weights of linear matrix of model plus one bias.
feval = function(x_new)
  if x ~= x_new then
    x:copy(x_new)
  end

  -- select a new training sample
  _nidx_ = (_nidx_ or 0) + 1
  if _nidx_ > (#data)[1] then
    _nidx_ = 1
  end

  local sample = data[_nidx_]
  local target = sample[{ {1} }] -- this syntax allows slicing arrays, result is torch.DoubleTensor of size 1
  local inputs = sample[{ {2,3} }] -- torch.DoubleTensor of size 2

  -- reset gradients (gradients are always accumulated, to accomodate batch methods)
  dl_dx:zero()

  local loss_x = criterion:forward(model:forward(inputs), target)
  model:backward(inputs, criterion: backward(model.output, target))
  -- backward propagation : http://neuralnetworksanddeeplearning.com/chap2.html

  return loss_x, dl_dx
end

sgd_params = {
  learningRate = 1e-3,
  learningRateDecay = 1e-4,
  weightDecay = 0,
  momentum = 0
}

for i=1, 1e4 do
  current_loss = 0

  for i=1,(#data)[1] do
    _, fs = optim.sgd(feval, x, sgd_params)
    -- feval : a closure that computes the loss and its gradient with regard to x, given a point x
    -- x is the vector of trainable weights

    current_loss = current_loss + fs[1]
  end

  current_loss = current_loss / (#data)[1]
  print('loss : ' .. current_loss)
end

text = {40.32, 42.92, 45.33, 48.85, 52.37, 57, 61.82, 69.78, 72.19, 79.42}

print('id','approx','text')
for i = 1,(#data)[1] do
  local myPrediction = model:forward(data[i][{{2,3}}])
  print(string.format("%2d  %6.2f %6.2f", i, myPrediction[1], text[i]))
end

-- least square solution (optimal solution)
size = data:size()
size[2] = size[2] + 1
newData = torch.Tensor(size):fill(1)
newData:narrow(2, 1, data:size()[2]):copy(data)
X = newData:narrow(2,2,3)
Y = newData:narrow(2,1,1)

theta = torch.inverse(X:transpose(1,2)*X)*X:transpose(1,2)*Y
theta:resize(3)

print('id','approx','text')
for i = 1,(#data)[1] do
  local dataWithBias = X[i]
  local myPrediction = dataWithBias*theta
  print(string.format("%2d  %6.2f %6.2f", i, myPrediction, text[i]))
end

dataTest = torch.Tensor{
  {6, 4},
  {10, 5},
  {14, 8}
}

for i = 1,(#dataTest)[1] do
  local prediction = model:forward(dataTest[i])
  print(string.format("{%2d, %2d} : %6.2f", dataTest[i][1], dataTest[i][2], prediction[1]))
end


-- 1e4
-- id approx  text
-- 1   40.09  40.32
-- 2   42.76  42.92
-- 3   45.21  45.33
-- 4   48.78  48.85
-- 5   52.34  52.37
-- 6   57.02  57.00
-- 7   61.92  61.82
-- 8   69.94  69.78
-- 9   72.39  72.19
-- 10   79.74  79.42

-- 1e5
-- id  approx   text
--  1   40.32  40.32
--  2   42.91  42.92
--  3   45.32  45.33
--  4   48.84  48.85
--  5   52.37  52.37
--  6   57.01  57.00
--  7   61.82  61.82
--  8   69.81  69.78
--  9   72.22  72.19
--  10   79.44  79.42
