require 'nn'
require 'cunn'

function create_model(opt)
  local n_inputs = 4
  local embedding_dim = 2
  local n_classes = 3

  -- linear -> sigmoid/requ -> linear -> softmax
  local model = nn.Sequential()
  model:add(nn.Linear(n_inputs, embedding_dim))

  if opt.nonlinearity_type == 'requ' then
    model:add(nn.ReQU())
  elseif opt.nonlinearity_type == 'sigmoid' then
    model:add(nn.Sigmoid())
  else
    error("requ or sigmoid is needed")
  end

  model:add(nn.Linear(embedding_dim, n_classes))
  model:add(nn.LogSoftMax())

  local criterion = nn.ClassNLLCriterion()

  return model, criterion
end

return create_model
