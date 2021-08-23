# what is this project about?

This is a lua library written in C++ that lets you load and call TorchScript models. 
It's kept in mind for the use with the [LÖVE](https://love2d.org/) engine to explore computer vision and machine learning techniques in games.

# How to build?

These instructions are only valid for Windows. If you're on Linux or Mac you'll have to figure it out yourself.

Prerequisites:
- [Python](https://www.python.org/)
- [Luajit](https://github.com/LuaJIT/LuaJIT) (compiled, after running the msvcbuild.bat file)
- [PyTorch + LibTorch (C++ library, Release Version)](https://pytorch.org/get-started/locally/). Add `libtorch/lib` to your path, or include the dlls with your Lua script or LÖVE program.
- Depending on the chosen LibTorch version you may also have to install [CUDA](https://developer.nvidia.com/cuda-zone) and [cudnn](https://developer.nvidia.com/cudnn)

```bash
cd build
cmake -G "Visual Studio 16 2019" -A x64 -DLUAJIT_PATH=/path/to/luajit -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
```

Now open the generated lamp.sln file and build it in Visual Studio.
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
    return lamp.newTensor(imageData:getPointer(), imageData:getSize(), imageData:getWidth(), imageData:getHeight(), imageData:getFormat())
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
