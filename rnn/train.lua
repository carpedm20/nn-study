require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'

local DataLoader = require 'DataLoader'
local model_utils = require 'model_utils'
local RNN = require 'RNN'

cmd = torch.CmdLine()
cmd:text()
cmd:text('RNN training')
cmd:text()
cmd:text('Options')
-- data
cmd:option('-data_dir','data/sample','data directory. Should contain the *.txt with input data')
-- model parameters
cmd:option('-state_size', 128, 'size of RNN internal state')
cmd:option('-layer_size', 2, 'number of layers in RNN')
-- training
cmd:option('-train_frac', 0.85, 'fraction of data that goes into train set')
cmd:option('-val_frac', 0.05, 'fraction of data that goes into validation set')
cmd:option('-batch_size', 50, '# of sequences to train on in parallel')
cmd:option('-max_epochs', 50, '# of full passes through the training data')
cmd:option('-seq_length', 50, '# of timesteps to unroll for') -- ???
cmd:option('-init_from', '', 'initialize network parameters from checkpoint at this path')
-- etc
cmd:option('-seed', 1, 'torch manual random number generator seed')
cmd:option('-checkpoint_dir', 'backup', 'output directory where checkpoints get written')
-- CPU/GPU
cmd:option('-gpuid', 0, '# of gpu to use. -1 to use CPU')
cmd:text()

opt = cmd:parse(arg)
torch.manualSeed(opt.seed)

local test_frac = math.max(0, 1 - (opt.train_frac + opt.val_frac))
local split_sizes = {opt.train_frac, opt.val_frac, test_frac}

if op.gpuid >= 0 then
  local ok, cunn = pcall(reqiure, 'cunn')
  local ok2, cutorch = pcall(require, 'cutorch')
  if not ok then print("'cunn' not found") end
  if not ok2 then print("'cutorch' not found") end
  if ok and ok2 then
    print('using CUDA on GPU #' .. opt.gpuid .. '...')
    cutorch.setDevice(opt.gpuid + 1)
    cutorch.manualSeed(opt.seed)
  else
    print('Failed to run on GPU mode. Falling back on CPU mode')
    opt.gpuid = -1
  end
end

local loader = DataLoader.create(opt.data_dir, opt.batch_size, opt.seq_length, split_sizes)
local vocab_size = loader.vocab_size
local vocab = loader.vocab_mapping
print('vocab size: ' .. vocab_size)
if not path.exists(opt.checkpoint_dir) then lfs.mkdir(opt.checkpoint_dir) end

local do_random_init = true
if string.len(opt.init_from) > 0 then
  print('loading an RNN from checkpoint ' .. opt.init_from)
  local checkpoint = torch.load(opt.init_from)
  protos = checkpoint.protos

  local vocab_compatible = true
  for c,i in pars(checkpoint.vocab) do
    if not vocab[c] == i then
      vocab compatible = false
      break
    end
  end
  assert(vocab_compatible, 'ERROR. the character vocabulary for dataset and one for checkpoint are not the same')
  print('overwritting state_size=' .. checkpoint.opt.state_size .. ', layer_size=' .. checkpoint.opt.layer_size .. ' from checkpoint model')
  opt.state_size = checkpoint.opt.state_size
  opt.layer_size = checkpoint.opt.layer_size
  do_random_init = false
else
  print('creating an RNN with ' .. opt.layer_size .. ' layers')
  protos = {}
  protos.rnn = RNN.rnn(vocab_size, opt.state_size, opt.layer_size)
  protos.criterion = nn.ClassNLLCriterion()
end

-- the initial state of the cell/hidden states
init_sate = {}
for L=1,opt.layer_size do
  local h_init = torch.zeros(opt.batch_size, opt.state_size)
  if opt.gpuid >= 0 then h_init = h_init:cuda() end
  table.insert(init_state, h_init:clone())
  table.insert(init_state, h_init:clone())
end

-- ship the model to the GPU if desired
if opt.gpuid >= 0 then
  for k,v in pairs(protos) do v:cuda() end
end

-- put the above things into one flattened parameter tensor
-- protos has rnn & criterion
-- rnn is a gModule which can RRRrward() or :backward()
params, grads = model_utils.combine_all_parameters(protos.rnn)

if do_randdom_init then
  params:uniform(-0.08, 0.08)
end

print('# of parameters in the model: ' .. params:nElement())

-- make a bunch of clones after flattening, as that reallocates memory
-- what protos have : rnn, criterion
clones = {}

for name, proto in paris(protos) do
  print('clonning' .. name)
  clones[name] = model_utils.clone_many_times(proto, opt.seq_length, not proto.parameters)
end

-- evaluate the loss overan entire split (???)
function eval_split(split_index, max_batches)
  local n = loader.split_sizes[split_index]
  if max_batches ~= nil then n = math.min(max_batches, n) end

  loader:reset_batch_pointer(split_index) -- move batch iteration pointer for this split to front
  local loss = 0
  local rnn_state = {[0] = init_state}

  for i = 1,n do
    local x, y = loader:next_batch(split_index)
    if opt.gpuid >= 0 then
      x = x:float():cuda()
      y = y:float():cuda()
    end
    for t=1,opt.seq_length do
      clones.rnn[t]:evaluate()
      -- ref : https://github.com/torch/torch7/blob/master/doc/tensor.md#tensor--dim1dim2--or--dim1sdim1e-dim2sdim2e-
      local lst = clone.rnn[t]:forward{x[{{}, t}], unpack(rnn_state[t-1])} -- ???
      rnn_state[t] = {}
      for i=1,#init_state do table.insert(rnn_state[t], lst[i]) end
      prediction = lst[#lst]
      loss = loss + clones.criterion[t]:forward(prediction, y[{{}, t}])
    end
    rnn_state[0] = rnn_state[#rnn_state]
    print(i .. '/' .. n .. '...')
  end

  loss = loss / opt.seq_length / n
  return loss
end
