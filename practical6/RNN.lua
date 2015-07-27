local RNN = {}

function RNN.rnn(opt)
  local x = nn.Identity()()
  local prev_h = nn.Identity()()

  function new_input_sum()
    local i2h = nn.Linear(opt.rnn_size, opt.rnn_size)(x)
    local h2h = nn.Linear(opt.rnn_size, opt.rnn_size)(prev_h)
    return nn.CAddTable()({i2h, h2h})
  end

  local nex_h = nn.Sigmoid()(new_input_sum())

  return nn.gMoudle({x, prev_h}, {next_h})
end

return RNN
