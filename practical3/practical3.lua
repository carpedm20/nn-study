require 'torch'
require 'math'
require 'nn'
require 'cunn'
require 'optim'
require 'gnuplot'
require 'dataset-mnist'

torch.manualSeed(1) -- fix random seed

-- TODO: play with these optimizer options for the second handin item, as described in the writeup
-- NOTE: see below for optimState, storing optimiser settings

local opt={}
opt.optimization = 'sgd'
opt.batch_size = 3
opt.train_size = 8000
opt.test_size = 0
opt.epochs = 2
opt.learningRate = 1e-1

local optimState
local optimMethod

-- optimMethods will return (new_parameters, table), where table[0] is the value of the function being optimized

if opt.optimization == 'lbfgs' then
  optimState = {
    learningRate = opt.learningRate,
    maxIter = 2,
    nCorrection = 10
  }
  optimMethod = optim.lbfgs
elseif opt.optimization == 'sgd' then
  optimState = {
    learningRate = opt.learningRate,
    weightDecay = 0,
    momentum = 0,
    learningRateDecay = 1e-7
  }
  optimMethod = optim.sgd
elseif opt.optimization == 'adagrad' then
  optimState = {
    learningRate = opt.learningRate
  }
  optimMethod = optim.adagrad
else
  error('Unknown optimizer: not lbfgs, sgd or adagrad')
end

-- mnist.download()

local function load_dataset(train_or_test, count)
  local data
  if train_or_test == 'train' then
    data = mnist.loadTrainSet(count, {32, 32})
  else
    data = mnist.loadTestSet(count, {32, 32})
  end

  local shuffled_indices = torch.randperm(data.data:size(1)):long()
  data.data = data.data:index(1, shuffled_indices):squeeze()
  data.labels = data.labels:index(1, shuffled_indices):squeeze()

  print('--------------------------------')
  print(' loaded dataset "' .. train_or_test .. '"')
  print('inputs', data.data:size())
  print('targets', data.labels:size())
  print('--------------------------------')

  -- vectorize each 2D to 1D
  data.data = data.data:reshape(data.data:size(1), 32*32)
  return data
end

train = load_dataset('train', opt.train_size)
test = load_dataset('test', opt.test_size)

n_train_data = train.data:size(1) -- 8000
n_test_data = test.data:size(1) -- 10000
n_inputs = train.data:size(2) -- 1024 = 32*32
n_outputs = train.labels:max()

print(train.labels:max())
print(train.labels:min())

lin_layer = nn.Linear(n_inputs, n_outputs)
softmax = nn.LogSoftMax()
model = nn.Sequential()
model:add(lin_layer)
model:add(softmax)

criterion = nn.ClassNLLCriterion()
params, gradParams = model:getParameters()

counter = 0
local feval = function(x)
  if x ~= params then
    params:copy(x)
  end
  
  start_idx = counter * opt.batch_size + 1
  end_idx = math.min(n_train_data, (counter+1) * opt.batch_size + 1)
  if end_idx == n_train_data then
    counter = 0
  else
    counter = counter + 1
  end

  -- grabs the next minibatch
  local batch_inputs = train.data[{{start_idx, end_idx}, {}}]
  local batch_targets = train.labels[{{start_idx, end_idx}}]
  gradParams:zero()

  local batch_outputs = model:forward(batch_inputs)
  local batch_loss = criterion:forward(batch_outputs, batch_targets)
  -- compute the derivative of the loss wrt the outputs of the model
  local dloss_doutput = criterion:backward(batch_outputs, batch_targets)
  -- use gradients to update weights
  model:backward(batch_inputs, dloss_doutput)

  -- returns the loss value and the gradient of the loss with regards to the parameters, evaluated on the minibatch.
  -- the gradient will be used to adjust the parameters so as to reduce the loss.
  return batch_loss, gradParams
end

losses = {}
epochs = opt.epochs -- number of full passes over all the training data
iterations = epochs * math.ceil(n_train_data / opt.batch_size)

--[[ optim method interface:

x*, {f}, ... = optim.method(func, x, state)
where:

func: a user-defined closure that respects this API: f, df/dx = func(x)
x: the current parameter vector (a 1D torch.Tensor)
state: a table of parameters, and state variables, dependent upon the algorithm
x*: the new parameter vector that minimizes f, x* = argmin_x f(x)
{f}: a table of all f values, in the order they've been evaluated (for some simple algorithms, like SGD, #f == 1)
]]

for i = 1, iterations do
  -- optimMethod is a variable storing a function, either sgd or adagrad ...
  local _, minibatch_loss = optimMethod(feval, params, optimState)
  -- the loss is "loss per data sample" because it is devided by the number of data potins.
  -- Since we evaluate the loss on a different minibatch each time, the loss will sometimes fluctuate upwards slightly.
  if i % 10 == 0 then
    print(string.format("batch processed: %6s, loss = %6.6f", i, minibatch_loss[1]))
  end
  losses[#losses + 1] = minibatch_loss[1]
end

test_inputs = test.data
test_targets = test.labels

n_correct = 0
n_false = 0

for i = 1, n_test_data do
  local prediction = model:forward(test_inputs[i])
  index = -1
  max = -9999
  for j = 1, 10 do
    if prediction[j] > max then
      index = j
    end
  end
  if index == test_targets[i] then
    n_correct = n_correct + 1
  else
    n_false = n_false + 1
  end
  if i % 500 == 0 then
    print(string.format("correct : %4s, false : %4s,", n_correct, n_false))
  end
end
