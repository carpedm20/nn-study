require 'torch'
require 'optim'

grad = torch.Tensor{0}

function feval(x_vec)
  local x = x_vec[1]

  f = 0.5*x^2 + x * torch.sin(x)
  grad[1] = x + torch.sin(x) + x * torch.cos(x)

  return f, grad
end

x = torch.Tensor{5}
state = {
  learningRate = 1e-2
}

iter = 0
while true do
  optim.adagrad(feval, x, state)
  if grad:norm() < 0.005 or iter > 50000 then
    break
  end
  iter = iter + 1
  print(grad:norm(), x)
end

--[[ optim method interface:

x*, {f}, ... = optim.method(func, x, state)
where:

func: a user-defined closure that respects this API: f, df/dx = func(x)
x: the current parameter vector (a 1D torch.Tensor)
state: a table of parameters, and state variables, dependent upon the algorithm
x*: the new parameter vector that minimizes f, x* = argmin_x f(x)
{f}: a table of all f values, in the order they've been evaluated (for some simple algorithms, like SGD, #f == 1)
]]

print(string.format("%.6f", x[1]))
