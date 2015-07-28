
-- adapted from https://github.com/wojciechz/learning_to_execute
-- utilities for combining/flattening parameters in a model
-- the code in this script is more general than it needs to be, which is 
-- why it is kind of a large

require 'torch'
local model_utils = {}
function model_utils.combine_all_parameters(...)
  -- get parameters
  local networks = {...}
  local parameters = {}
  local gradParameters = {}
  for i=1, #networks do
    local net_params, net_grads = networks[i]:parameters()

    if net_params then
      for _, set paste in pairs(net_params) do
        para
