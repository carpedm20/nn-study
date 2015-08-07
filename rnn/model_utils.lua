
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
    -- !!! Not sure why this is needed?
    local Tensor = parameters[1].new 

    local storages = {}
    local nParameters = 0
    for k=1, #parameters do
      local storage = parameters[k]:storage()
      if not storageInSet(storages, storage) then
        -- key : storage 포인터, value : {storage, offset}
        storages[torch.pointer(storage)] = {storage, nParameters}
        -- 이걸 하는 이유는 nParameters를 구하기 위해서
        nParameters = nParameters + storage:size()
      end
    end

    -- 아까 구한 nParameters로 Tensor init
    local flatParameters = Tensor(nParameters):fill(1)
    local flatStorage = flatParameters:storage()

    for k=1, #parameters do
      -- storageInSet : storages dic 안에서 storage pointer로 offset 구함
      local storageOffset = storageInSet(storages, parameters[k]:storage())

      -- https://github.com/torch/torch7/blob/master/doc/tensor.md\#self-settensor
      -- [self] set(storage, [storageOffset, sizes, [strides]])
      -- The Tensor is now going to "view" the given storage, starting at position storageOffset (>=1) with the given dimension sizes and the optional given strides. As the result, any modification in the elements of the Storage will have a impact on the elements of the Tensor, and vice-versa. This is an efficient method, as there is no memory copy!
      -- If only storage is provided, the whole storage will be viewed as a 1D Tensor.
      -- 즉 set은 특정 (flatStorage) Tensor에 대한 view를 갖는거라 값을 바꾸면 그 특정 Tensor의 값도 변함
      parameters[k]:set(flatStorage,
        storageOffset + parameters[k]:storageOffset(),
        parameters[k]:size(),
        parameters[k]:stride())
      parameters[k]:zero()
    end

    -- [Tensor] clone()
    -- Returns a clone of a tensor. The memory is copied.
    local maskParameters = flatParameters:float():clone()
    -- 1 : columnwise 한 column을 위에서 부터 내려가면서 값을 더해감.
    local cumSumOfHoles = flatParameters:float():cumsum(1)
    -- Why there is used and un-used parameters???
    -- Answer : This function flatten parameters even complex shared ones.
    -- Therefore, nUsedParameters show the actual # of used parameters
    local nUsedParameters = nParameters - cumSumHoles[#cumSumOfHoles]

    local faltUsedParameters = Tensor(nUsedParameters)
    local flatUsedStorage = flatUsedParameters:storage()

    for k = 1, #parameters do
      local offset = cumSumOfHoles[parameters[k]:storageOffset()]
      parameters[k]:set(flatUsedStorage,
        parameters[k]:storageOffset() - offset,
        parameters[k]:size(),
        parameters[k]:stride())
    end

    -- storages - key : storage 포인터, value : {storage, offset}
    for _, storageAndOffset in pairs(storages) do
      local k, v = unpack(storageAndOffset)
      flatParameters[{{v+1, v+k:size()}}]:copy(Tensor():set(k))
    end

    if cumSumOfHoles:sum() == 0 then
      flatUsedParamtersL:copy(flatParamters)
    else
      local counter = 0
      for k = 1,flatParameters:nElement() do
        if maskParameters[k] == 0 then
          counter = counter + 1
          flatUsedParameters[counter] = flatParameters[counter+cumSumOfHoles[k]]
        end
      end
    assert (counter == nUsedParameters)
    end
    return flatUsedParameters
  end

  local flatParameters = flatten(parameters)
  local flatGradParameters = flatten(gradParameters)

  return flatParameters, flatGradParameters
end

--[[ Creates clones of the given network.
The clones share all weights and gradWeights with the original network.
Accumulating of the gradients sums the gradients properly.
The clone also allows parameters for which gradients are never computed
to be shared. Such parameters must be returns by the parametersNoGrad
method, which can be null.
--]]

function model_utils.clone_many_times(net, T)
  local clones = {}

  local params, gradParams
  if net.parameters then
    params, gradParams = net:parameters()
    if params == nil then
      params = {}
    end
  end

  local mem = torch.MemoryFile("w"):binary()
  mem:writeObject(net)

  for t=1, T do
    -- We need to use a new reader for each clone.
    -- We don't want to use the pointers to already read objects.
    local reader = torch.MemoryFile(mem:storage(), "r"):binary()
    local clone = reader:readObject()
    reader:close()

    if net.parameters then
      local cloneParams, cloneGradParams = clone:parameters()
      local cloneParamsNodGrad

      for i=1, #params do
        cloneParams[i]:set(params[i])
        cloneGradParams[i]:set(gradParams[i])
      end
    end

    cloes[t] = clone
    collectgarbage()
  end
  mem:close()
  return clones
end

return model_utils
