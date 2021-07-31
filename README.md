# what is this project about?

This is a lua library written in c++ that lets you load and call TorchScript models. It's kept in mind for the use with the LÖVE engine to explore computer vision techniques in games.

# How to build?

I can only tell you how to do it on windows, if you're on linux or mac you'll have to figure it out yourself

Prerequisites:
- Luajit (already compiled)
- libtorch

```bash
cd build
cmake -G "Visual Studio 16 2019" -A x64 -DLUAJIT_PATH=/path/to/luajit -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
cmake --build . --config Release
```

The compiled `lamp.dll` will be found in the `Release` folder.

# How to use?

First create a model in python and script + save it:

```python
import torch


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.lin = torch.nn.Linear(80 * 80 * 4, 1)

    def forward(self, x):
        x = torch.flatten(x)
        return self.lin(x)


sm = torch.jit.script(Model())
sm.save("model.pt")
```

Then you can do the following within löve
```lua
local lamp = require "lamp"

local model
local canvas
local tensor

local function createTensor(imageData)
    return lamp.newTensor(imageData:getPointer(), imageData:getWidth(), imageData:getHeight(), imageData:getFormat())
end

function love.load()
    model = lamp.newModule("model.pt")

    local length = 80
    canvas = love.graphics.newCanvas(length, length, {format="rgba32f"})
    canvas:renderTo(function()
        love.graphics.setColor(0, 0, 1, 1)
        love.graphics.rectangle("fill", 0, 0, length / 2, length / 2)
        love.graphics.setColor(0, 1, 0, 1)
        love.graphics.rectangle("fill", length / 2, 0, length / 2, length / 2)
        love.graphics.setColor(1, 0, 0, 1)
        love.graphics.rectangle("fill", 0, length / 2, length / 2, length / 2)
    end)

    local imageData = canvas:newImageData()
    tensor = createTensor(imageData)
    local output = model:forward(tensor)
end

function love.draw()
    love.graphics.setColor(1, 1, 1, 1)
    love.graphics.print(tostring(love.timer.getFPS()))
    love.graphics.circle("line", 100, 100, 100)
end
```
