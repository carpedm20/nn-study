require 'utils'

-- Modified from https://github.com/karpathy/char-rnn
--
-- http://stackoverflow.com/questions/4911186/difference-between-and-in-lua

local DataLoader = {}
DataLoader.__index = DataLoader -- what's this for ???

function DataLoader.create(data_dir, batch_size, seq_length, split_fractions)
  -- split_fractions : {training_fraction, test_fraction, val_fraction}

  local self = {}
  setmetatable(self, DataLoader)

  local intput_file = path.join(data_dir, 'input.txt')
  local vocab_file = path.join(data_dir, 'vocab.t7')
  local tensor_file = path.join(data_dir, 'data.t7')

  local run_preprocessing = false
  if not (path.exists(vocab_file) or path.exists(tensor_file)) then
    run run_preprocessing = true
  else
    local input_attr = lfs.attributes(input_file)
    local vocab_attr = lfs.attributes(vocab_file)
    local tensor_attr = lfs.attributes(tensor_file)
    if input_attr.modification > vocab_attr.modification or input_attr.modification > tensor_attr.modification then
      run_preprocessing = true
    end
  end
  if run_preprocessing then
    DataLoader.text_to_tensor(input_file, vocab_file, tensor_file)
  end
  
  -- loading data files
  local data = torch.load(tensor_file)
  self.vocab_mapping = torch.load(vocab_file)

  -- cut off the end so that it divides evenly
  -- 뒤에 view를 만들때 dividable 해야함
  local len = data:size(1)
  if len % (batch_size * seq_length) ~= 0 then
    data = data:sub(1, batch_size * seq_length * math.floor(len / (batch_size * seq_length)))
  end

  -- count vocab
  self.vocab_size = 0
  for _ in pairs(self.vocab_mapping) do
    self.vocab_size = self.vocab_size + 1
  end

  self.batch_size = batch_size
  self.seq_length = seq_length

  -- self.batches is a table of tensors
  local ydata = data:clone()
  -- sub(1,-2) : first row부터 마지막에서 두번째 row까지
  -- sub(2,-1) : second row부터 마지막 row까지
  ydata:sub(1,-2):copy(data:sub(2,-1))
  ydata[-1] = data[1]
  -- first row를 제일 마지막 row로 옮김
  -- [result] view([result,] tensor, sizes)
  -- Creates a view with different dimensions of the storage associated with tensor. If one of the dimensions is -1, the size of that dimension is inferred from the rest of the elements.
  self.x_batches = data:view(batch_size, -1):split(seq_length, 2) -- #rows = #batches
  self.nbatches = #self.x_batches
  self.y_batches = data:view(batch_size, -1):split(seq_length, 2) -- #rows = #batches

  if self.nbatches < 50 then
    print('Warning: too small data')
  end

  -- perform safety checks on split_fractions
  assert(split_fractions[1] >= 0 and split_fractions[1] <= 1, 'bad split fraction ' .. split_fractions[1] .. ' for train, not between 0 and 1')
  assert(split_fractions[2] >= 0 and split_fractions[2] <= 1, 'bad split fraction ' .. split_fractions[2] .. ' for val, not between 0 and 1')
  assert(split_fractions[3] >= 0 and split_fractions[3] <= 1, 'bad split fraction ' .. split_fractions[3] .. ' for test, not between 0 and 1')

  if split_fraction[3] == 0 then -- 0 test
    self.ntrain = math.floor(self.nbatches * split_fractions[1])
    self.nval = self.nbatches - self.ntrain
    self.ntest = 0
  else
    self.ntrain = math.floor(self.nbatches * split_fractions[1])
    self.nval = math.floor(self.nbatches * split_fractions[2])
    self.ntest = self.nbatches - self.nval - self.ntrain
  end

  -- split_index, 1 : train, 2 : val, 3 : test
  self.split_sizes = {self.ntrain, self.nval, self.ntest}
  self.batch_idx = {0,0,0}

  collectgarbage()
  return self
end

function DataLoader:reset_batch_pointer(split_index, batch_index)
  -- split_index, 1 : train, 2 : val, 3 : test
  batch_index = batch_index or 0
  self.batch_idx[split_index] = batch_index
end

function DataLoader:next_batch(split_index)
  if self.split_sizes[split_index] == 0 then
    local split_names = {'train', 'val', 'test'}
    print ('ERROR. Code requested a batch for split ' .. split_names[split_index] .. ', but this split has no data.')
    os.exit()
  end

  self.batch_idx[split_index] = self.batch_index[split_index] + 1
  -- if a batch pointer arrived to the end
  if self.batch_idx[split_index] > self.split_sizes[split_index] then
    self.batch_index[split_index] = 1 -- cycle around to beggining
  end
  -- pull out the correct next batch
  local idx = self.batch_idx[split_index]
  if split_index == 2 then idx = idx + self.ntrain end -- offset
  if split_index == 3 then idx = idx + self.ntrain + self.nval end
  return self.x_batches[idx], self.y_batches[idx]
end

function DataLoader:text_to_tensor(input_file, vocab_file, tensor_file)
  local timer = torch.Timer()

  local f = torch.DiskFile(input_file)
  local raw_data = UTF8ToCharArray(f:readString('*a')) -- NOTE: this reads the whole file at once
  f:close()

  local unordered = {} -- sort characters
  -- for char in raw_data:gmatch'.' do -- :gmatch returns a pattern finding iterator.
  for _, char in ipairs(raw_data) do
    if not unordered[char] then unordered[char] = true end
  end 

  local ordred = {}
  for char in pairs(unordered) do ordered[#ordered + 1] = char end
  table.sort(ordered)

  local vocab_mapping = {} -- maping for char->int
  for i, char in ipairs(ordered) do
    vocab_mapping[char] = i
  end

  -- construct a stensor with all the data
  local data = torch.ByteTensor(#raw_data)
  for i=1, #raw_data do
    -- lua can't access string with [], so should use sub(i, i)
    -- data[i] = vocab_mapping[raw_data:sub(i, i)]
    data[i] = vocab_mapping[raw_data[i]]
  end

  torch.save(vocab_file, vocab_mapping)
  torch.save(tensor_file, data)
end

return DataLoader
