trainSet = {}
testSet = {}

function trainSet:size()
    return trainCount
end

-- Download data if not local.
if not paths.filep('iris.data') then
    print("Getting data...")
    data = "http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    os.execute('wget ' .. data)
end

-- Load data.
math.randomseed(os.time())
trainCount = 0; testCount = 0
file = io.open('iris.data')
for line in file:lines() do
    if (string.len(line) > 0) then

        -- Read line from file.
        x1, x2, x3, x4, species = unpack(line:split(","))
        input = torch.Tensor({x1, x2, x3, x4});
        output = torch.Tensor(3):zero();

        -- Set output based on species.
        if (species == "Iris-setosa") then
            output[1] = 1
        elseif (species == "Iris-versicolor") then
            output[2] = 1
        else
            output[3] = 1
        end

        -- Keep 20% of data aside for testing.
        if (math.random() > 0.2) then
            table.insert(trainSet, {input, output})
            trainCount = trainCount + 1
        else
            table.insert(testSet, {input, output})
            testCount = testCount + 1
        end

    end
end

-- Initialise the network.
require "nn"
inputs = 4; outputs = 3; hidden = 10;
mlp = nn.Sequential();
mlp:add(nn.Linear(inputs, hidden))
mlp:add(nn.Tanh())
mlp:add(nn.Linear(hidden, outputs))

-- Train the network.
criterion = nn.MSECriterion() 
trainer = nn.StochasticGradient(mlp, criterion)
trainer.learningRate = 0.01
trainer.maxIteration = 25
trainer:train(trainSet)

-- Test the network.
correct = 0
for i = 1, testCount do
    val = mlp:forward(testSet[i][1])
    out = testSet[i][2]
    z = val:add(out)
    if (torch.max(z) > 1.5) then
        correct = correct + 1
    end
end
print(string.format("%.2f%s", 100 * correct / testCount, "% correct"))
