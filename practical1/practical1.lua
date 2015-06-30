require('torch')

local t = torch.Tensor(6,6)
local t2 = torch.Tensor(6,6)


t3 = t+ t2
t:add(t2) -- don't need to allocate additional memory

print(t3)
print(t)

local t = torch.Tensor({{1,2,3},{4,5,6},{7,8,9}})

-- List 3 expressions that can replace the first line below to slice (extract) the middle column from t:

x = torch.Tensor(4,5,6,2,5) -- 4D tensor 4x5x6x2
print(x:nDimension())
print(x:size())
print(x[1][1][1][1][1])
print(x[1][1][1][1])

s = torch.LongStorage(6)
-- The actual data of a Tensor is contained into a Storage
s[1] = 4; s[2] = 5; s[3] = 6; s[4]=1; s[5]=4; s[6]=3;
x = torch.Tensor(s)

x = torch.Tensor(4,5)
-- The actual data of a Tensor is contained into a Storage
s = x:storage()
for i=1, s:size() do
  s[i] = i
  -- elements in the same row [elements along the last dimension] are contiguous in memory
end
print(x)

x = torch.Tensor(4,5,3)
print(x:stride(1)) -- 15=5*3
print(x:stride(2)) -- 3
print(x:stride(3)) -- 1

x = torch.Tensor(4,5)
i=0
x:apply(function()
  i=i+1
  return i
end)

print(x:type())

x = torch.Tensor(15):zero()
x:narrow(1,2,3):fill(1)

print(x)
print(x:narrow(1,2,3))
print(x:narrow(1,3,6))

x = torch.Tensor(5,5):zero()
x:narrow(2,1,2):fill(4)
print(x)
x = torch.Tensor(5,5):zero()
x:narrow(1,1,2):fill(4)
print(x)


x = torch.Tensor(2,2,2):zero()
x:narrow(2,1,1):fill(4)
print(x)
x = torch.Tensor(2,2,2):zero()
x:narrow(1,1,1):fill(4)
print(x)

print(x:size())
s = x:storage()
print(s:size())

y = torch.Tensor(s:size()):copy(x)
z = x:clone()
print(x==y)

t = torch.Tensor(10,10)
t2 = torch.Tensor(10,10)
t:narrow(1,5,3):fill(4)
t2:narrow(2,5,3):fill(4)
t3 = t + t2

print(t3)

t = torch.Tensor(3,3)
s = t:storage()
for i=1,s:size() do
  s[i] = i
end
print(t)

print(t[{2,{1,3}}])
print(t[1][1])

-- Expressions that can replace the first line below to slice (extract) the middle column from t:
col = t:narrow(2,2,1)
print(col)
col = t[{{},2}]
print(col)
col = t[{{1,3},2}]
print(col)

t[torch.lt(t,5)] = -2
print(t)

x = torch.rand(5,5)

