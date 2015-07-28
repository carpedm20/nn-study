local LSTM = {}

function LSTM.lstm(opt)
  local x = nn.Identity()()
  local prev_c = nn.Identity()()
  local prev_h = nn.Identity()()

  function new_input_sum()
    -- input to hidden
    local i2h = nn.Linear(opt.rnn_size, opt.rnn_size)(x)
    -- hidden to hidden
    local h2h = nn.Linear(opt.rnn_size, opt.rnn_size)(prev_h)
    return nn.CAddTable()({i2h, h2h})
  end

  local in_gate = nn.Sigmoid()(new_input_sum())
  local forget_gate = nn.Sigmoid()(new_input_sum())
  local out_gate = nn.Sigmoid()(new_input_sum())
  local in_transform = nn.Tanh()(new_input_sum())

  local next_c = nn.CAddTable()({
    nn.CMulTable()({forget_gate, prev_c}),
    nn.CMulTable()({in_gate, in_transform})
  })
  local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})

  return nn.gModule({x, prev_c, prev_h}, {next_c, next_h})
end

function LSTM.fast_lstm(input_size, rnn_size)
  local x = nn.Identity()()
  local prev_c = nn.Identity()()
  local prev_h = nn.Identity()()

  local i2h = nn.Linear(input_size, 4 * rnn_size)(x)
  local h2h = nn.Linear(rnn_size, 4 * rnn_size)(prev_h)
  local all_input_sums = nn.CAddTable()({i2h, h2h})

  local sigmoid_chunk = nn.Narrow(2, 1, 3 * rnn_size)(all_input_sums)
  sigmoid_chunk = nn.Sigmoid()(sigmoid_chunk)
  local in_gate = nn.Narrow(2, 1, rnn_size)(sigmoid_chunk)
  local forget_gate = nn.Narrow(2, rnn_size + 1, rnn_size)(sigmoid_chunk)
  local out_gate = nn.Narrow(2, 2 * rnn_size + 1, rnn_size)(sigmoid_chunk)

  local in_transform = nn.Narrow(2, 3 * rnn_size + 1, rnn_size)(all_input_sums)
  in_transform = nn.Tanh()(in_transform)

  local next_c           = nn.CAddTable()({
      nn.CMulTable()({forget_gate, prev_c}),
      nn.CMulTable()({in_gate,     in_transform})
    })
  local next_h           = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})

  return nn.gModule({x, prev_c, prev_h}, {next_c, next_h})
end

return LSTM
