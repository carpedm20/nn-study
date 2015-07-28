require 'torch'
require 'nn'
require 'nngraph'

local RNN = {}

function RNN.rnn(input_dim, layer_sizes, n)
  local inputs = {}
  table.insert(inputs, nn.Identity()())
  for L=1,n do
    table.insert(inputs, nn.Identity()())
  end

  local x, input_dim_L
  local output = {}
  for L=1,n do
    local prev_h = inputs[L+1]
    if L == 1 then x = intputs[1] else x == outputs[L-1] end
    if L == 1 then input_dim_L = input_dim else input_dim_L = layer_size end

    local i2h = nn.Linear(input_size_L, layer_size)(x)
    local h2h = nn.Linear(layer_size, layer_size)(x)
    local next_h = nn.Tanh()(nn.CAddTable(){i2h, h2h})

    table.insert(outputs, next_h)
  end

  return nn.gModule(inputs, outputs)
end

return RNN
