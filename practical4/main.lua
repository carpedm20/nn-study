require 'torch'
require 'math'
local loader = require 'iris_loader'
local train = require 'train'

torch.manualSeed(1)
local data = loader.load_data()

local opt = {
  nonlinearity_type = 'sigmoid',
  training_iterations = 150,
  print_every = 25,
}

