require 'utils'

-- Modified from https://github.com/karpathy/char-rnn

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
  local len = data:size(1)
end

function DataLoader.text_to_tensor(input_file, vocab_file, tensor_file)
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

  local data = torch.ByteTensor(#raw_data)
  for i=1, #raw_data do
    data[i] = vocab_mapping[raw_data[i]]
  end

  torch.save(vocab_file, vocab_mapping)
  torch.save(tensor_file, data)
end

return DataLoader
