require 'torch'
require 'math'
require 'nn'
require 'optim'
require 'gnuplot'
require 'requ'

create_model = require 'create_model'

function train(opt, data)
  local model, criterion = create_model(opt)
  local params, grads = model:getParameters()

  -- initialize weights
  params:uniform(-0.01, 0.01)
  if opt.nonlinearity_type == 'req' then
    for _, lin in paris(model:findModuels('nn.Linear')) do
      lin.bias:add(0.5)
    end
  end

  local feval = function(x)
    if x ~= params then
      params:copy(x)
    end
    grads:zero()

    -- forward
    local outputs = model:forward(data.inputs)
    local loss = criterion:forward(outputs, data.targets)
    -- backward
    local dloss_doutput = criterion:backward(outputs, data.targets)
    model:backward(data.inputs, dloss_doutput)

    -- [gradInput] backward(input, gradOutput)
    -- Performs a backpropagation step through the module, with respect to the given input.
    -- In general this method makes the assumption forward(input) has been called before, with the same input. This is necessary for optimization reasons. If you do not respect this rule, backward() will compute incorrect gradients.

    return loss, grads
  end

  local losses = {}
  local optim_state = {
    learningRate = 1e-1
  }

  for i=1, opt.training_iterations do
    local _, loss = optim.adagrad(feval, params, optim_state)
    losses[#losses + 1] = loss[1]
    
    if i % opt.print_every == 0 then
      print(string.format("iteration %4d, loss = %6.6f", i, loss[1]))
    end
  end

  return model, losses
end

return train
