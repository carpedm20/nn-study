local RNN = {}

function RNN.rnn(vocab_size, state_size, layer_size)
  local inputs = {}
  table.insert(inputs, nn.Identity()())
  for L = 1, layer_size do
    table.insert(inputs, nn.Identity()())
  end

  local x, input_size_L
  local outputs = {}
  for L = 1, layer_size do
    local prev_h = inputs[L+1]
    if L == 1 then x = inputs[1] else x = outputs[L-1] end
    if L == 1 then input_size_L = vocab_size else input_size_L = state_size end

    local i2h = nn.Linear(input_size_L, state_size)(x)
    local h2h = nn.Linear(state_size, state_size)(prev_h)
    local next_h = nn.Tanh()(nn.CAddTable()(i2h, h2h))

    table.insert(outputs, next_h)
  end

  return nn.gMoudle(inputs, outputs)
end

return RNN
