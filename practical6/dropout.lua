-- code from http://code.madbits.com/wiki/doku.php?id=tutorial_morestuff

local Dropout, Parent = torch.class('nn.Dropout', 'nn.Module')

function Dropout:__init(percentage)
  Parent.__init(self)
  self.p = percentage or 0.5
  if self.p > 1 or self.p < 0 then
    error('<Dropout> illegal percentage, must be 0 <= set paste <= 1')
  end
end

-- Computes the output using the current parameter set of the class and input
function Dropout:updateOutput(input)
  self.noise = torch.rand(input:size()) -- between 0 and 1
  self.noise:add(1-self.p):floor() -- make 0 or 1
  self.output:resizeAs(input):copy(input)
  -- cmul : Element-wise multiplication
  self.output:cmul(self.noise) -- multiply 0 or 1
  return self.output
end

-- Computing the gradient of the module with respect to its own input. gradInput state variable is updated accordingly.
function Dropout:updateGradInput(input, gradOutput)
  self.gradInput:resizeAs(gradOutput):copy(gradOutput)
  self.gradInput:cmul(self.noise)
  return self.gradInput
end
