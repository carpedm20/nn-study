
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
      for _, p in pairs(net_params) do
        parameters[#parameters + 1] = p
      end
      for _, g in pairs(net_grads) do
        gradParameters[#gradParameters + 1] = g
      end
    end
  end

  local function storageInSet(set, storage)
    -- [number] torch.pointer(object)
    -- Returns a unique id (pointer) of the given object, which can be a Torch object, a table, a thread or a function.
    local storageAndOffset = set[torch.pointer(storage)]
    if storageAndOffset == nil then
      return nil
    end
    -- storages[torch.pointer(storage)] = {storage, nParameters}
    local _, offset = unpack(storageAndOffset)
    return offset
  end

  -- flattens arbitrary lists of parameters.
  local function flatten(parameters)
    if not parameters or #parameters == 0 then
      return torch.Tensor()
    end
    -- To begin flattening, start with first parameter
    -- Initialize Tensor with the same size with first parameter
    local Tensor = parameters[1].new 

    local storages = {}
    local nParameters = 0
    for k=1, #parameters do
      local storage = parameters[k]:storage()
      if not storageInSet(storages, storage) then
        storages[torch.pointer(storage)] = {storage, nParameters}
        nParameters = nParameters + storage:size()
      end
    end

    local flatParameters = Tensor(nParameters):fill(1)
    local flatStorage = flatParameters:storage()

    for k=1,#parameters do
      local storageOffset = storageInSet(storages, parameters[k]:storage())
      -- https://github.com/torch/torch7/blob/master/doc/tensor.md\#self-settensor
      parameters[k]:set(
