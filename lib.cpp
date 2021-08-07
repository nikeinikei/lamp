#include "lua.hpp"
#include "torch/script.h"
#include "torch/optim.h"

#include <atomic>
#include <cstring>
#include <exception>
#include <functional>
#include <memory>
#include <string>

enum class ObjectType {
    MODULE,
    TENSOR,
    UNKNOWN,
};

class Object {
public:
    virtual const char* getName() {
        return "lamp.Object";
    }

    virtual ObjectType getType() {
        return ObjectType::UNKNOWN;
    }

    void retain() {
        count.fetch_add(1, std::memory_order_relaxed);
    }

    void release() {
        if (count.fetch_sub(1, std::memory_order_release) == 1) {
            std::atomic_thread_fence(std::memory_order_acquire);
            delete this;
        }
    }

private:
    std::atomic<int> count = 1;

};

template <typename T>
T* luax_checktype(lua_State* L, int idx, ObjectType type) {
    Proxy* u = (Proxy*) lua_touserdata(L, idx);
    if (u->obj == nullptr) {
        luaL_error(L, "cannot use object after it has been released");
    }
    return (T*)u->obj;
}

static void luax_catchexcept(lua_State* L, std::function<void()> closure) {
    try {
        closure();
    }
    catch (std::exception e) {
        luaL_error(L, e.what());
    }
}

struct Proxy {
    Object* obj;
    ObjectType type;
};

static int w__gc(lua_State* L) {
    Proxy* p = (Proxy*) lua_touserdata(L, 1);
    if (p->obj != nullptr) {
        p->obj->release();
        p->obj = nullptr;
    }

    return 0;
}

static luaL_Reg base_methods[] = {
    { "__gc", w__gc },
    { 0, 0 }
};

static void luax_register_methods(lua_State* L, luaL_Reg* methods) {
    int i = 0;
    while (1) {
        luaL_Reg method = methods[i];
        if (method.name == 0) {
            break;
        }

        lua_pushcfunction(L, method.func);
        lua_setfield(L, -2, method.name);

        i++;
    }
}

static void luax_registertype(lua_State* L, const char* name, luaL_Reg *methods) {
    luaL_newmetatable(L, name);

    luax_register_methods(L, base_methods);
    luax_register_methods(L, methods);

    lua_pushvalue(L, -1);
    lua_setfield(L, -2, "__index");

    lua_pop(L, 1);
}

static void luax_pushtype(lua_State* L, Object* obj) {
    obj->retain();

    Proxy* proxy = (Proxy*) lua_newuserdata(L, sizeof(Proxy));
    proxy->obj = obj;
    proxy->type = obj->getType();

    lua_getfield(L, LUA_REGISTRYINDEX, obj->getName());
    lua_setmetatable(L, -2);
}

static void charArrayDeleter(void* data) {
    char* x = (char*) data;
    delete x;
}

static void floatArrayDeleter(void* data) {
    float* x = (float*) data;
    delete x;
}

class TensorWrapper : public Object {
public:
    constexpr static const char* name = "lamp.Tensor";

    TensorWrapper(void* data, size_t numBytes, int width, int height, const char* format) {
        int64_t numComponents = 0;
        at::TensorOptions options;
        if (strcmp(format, "rgba32f") == 0) {
            options = options.dtype(torch::kFloat32);
            numComponents = 4;
        }
        else if (strcmp(format, "rg32f") == 0) {
            options = options.dtype(torch::kFloat32);
            numComponents = 2;
        }
        else if (strcmp(format, "r32f") == 0) {
            options = options.dtype(torch::kFloat32);
            numComponents = 1;
        }
        else if (strcmp(format, "rgba16f") == 0) {
            options = options.dtype(torch::kFloat16);
            numComponents = 4;
        }
        else if (strcmp(format, "rg16f") == 0) {
            options = options.dtype(torch::kFloat16);
            numComponents = 2;
        }
        else if (strcmp(format, "r16f") == 0) {
            options = options.dtype(torch::kFloat16);
            numComponents = 1;
        }
        if (numComponents > 0) {
            char* copy = new char[numBytes];
            std::memcpy(copy, data, numBytes);
            tensor = torch::from_blob(copy, { width, height, numComponents }, charArrayDeleter, options);
        }
        else {
            throw std::exception("unsupported format");
        }
    }

    TensorWrapper(float* values, size_t numValues) {
        at::TensorOptions options;
        options = options.dtype(torch::kFloat);
        tensor = torch::from_blob(values, { (int64_t) numValues }, floatArrayDeleter, options);
    }

    TensorWrapper(const at::Tensor& t) {
        tensor = t;
    }

    TensorWrapper* toDevice(const char* identifier) const {
        return new TensorWrapper(tensor.to(c10::Device(identifier)));
    }

    TensorWrapper* cpu() {
        return new TensorWrapper(tensor.to(c10::Device(c10::DeviceType::CPU, 0)));
    }

    TensorWrapper* cuda() {
        return new TensorWrapper(tensor.to(c10::Device(c10::DeviceType::CUDA, -1)));
    }

    const at::Tensor& getTensor() const {
        return tensor;
    }

    const char* getName() override {
        return TensorWrapper::name;
    }

    ObjectType getType() override {
        return ObjectType::TENSOR;
    }

    double item() {
        return tensor.item<double>();
    }

    void backward() {
        try {
            tensor.backward();
        } catch(c10::Error& err) {
            throw std::exception(err.what());
        }
    }

private:
    at::Tensor tensor;
};

static TensorWrapper* luax_checktensor(lua_State* L, int idx) {
    return luax_checktype<TensorWrapper>(L, idx, ObjectType::TENSOR);
}

static int w_tensor_backward(lua_State* L) {
    TensorWrapper* self = luax_checktensor(L, 1);
    luax_catchexcept(L, [&]() { self->backward(); });

    return 1;
}

static int w_tensor_item(lua_State* L) {
    TensorWrapper* self = luax_checktensor(L, 1);
    double value = self->item();
    lua_pushnumber(L, value);

    return 1;
}

static int w_tensor_cuda(lua_State* L) {
    TensorWrapper* self = luax_checktensor(L, 1);
    TensorWrapper* newTensor = self->cuda();
    luax_pushtype(L, newTensor);
    newTensor->release();

    return 1;
}

static int w_tensor_cpu(lua_State* L) {
    TensorWrapper* self = luax_checktensor(L, 1);
    TensorWrapper* newTensor = self->cpu();
    luax_pushtype(L, newTensor);
    newTensor->release();

    return 1;
}

static int w_tensor_toDevice(lua_State* L) {
    TensorWrapper* self = luax_checktensor(L, 1);
    const char* deviceIdentifier = luaL_checkstring(L, 2);
    TensorWrapper* newTensor = self->toDevice(deviceIdentifier);
    luax_pushtype(L, newTensor);
    newTensor->release();

    return 1;
}

struct luaL_Reg tensor_functions[] = {
    { "backward", w_tensor_backward },
    { "cuda", w_tensor_cuda },
    { "cpu", w_tensor_cpu },
    { "toDevice", w_tensor_toDevice },
    { "item", w_tensor_item },
    { 0, 0 }
};

class ModuleWrapper : public Object {
public:
    constexpr static const char const* name = "lamp.Module";

    ModuleWrapper(const char* path) {
        try {
            mod = torch::jit::load(path);
            std::vector<at::Tensor> parameters;
            for (auto& param : mod.parameters()) {
                parameters.push_back(param);
            }
            opt = std::make_unique<torch::optim::Adam>(parameters);
        }
        catch (c10::Error e) {
            throw std::exception(e.what());
        }
    }

    void cuda() {
        mod.to(c10::Device(c10::DeviceType::CUDA, -1));
    }

    void cpu() {
        mod.to(c10::Device(c10::DeviceType::CPU, 0));
    }

    void toDevice(const char* deviceIdentifier) {
        mod.to(c10::Device(deviceIdentifier));
    }

    const char* getName() override {
        return ModuleWrapper::name;
    }

    ObjectType getType() override {
        return ObjectType::MODULE;
    }

    TensorWrapper* forward(std::vector<c10::IValue> inputs) {
        try {
            auto out = mod.forward(inputs);
            if (out.isNone()) {
                return nullptr;
            } else {
                return new TensorWrapper(out.toTensor());
            }
        }
        catch (c10::Error err) {
            throw std::exception(err.what());
        }
    }

    void zeroGrad() {
        opt->zero_grad();
    }

    void step(double learningRate) {
        torch::NoGradGuard no_grad;
        for (auto& param : mod.parameters()) {
            param.subtract_(param.grad(), learningRate);
        }
    }

    void save(const char* path) {
        mod.save(path);
    }

private:
    torch::jit::script::Module mod;
    std::unique_ptr<torch::optim::Adam> opt;
};

static ModuleWrapper* luax_checkmodule(lua_State* L, int idx) {
    return luax_checktype<ModuleWrapper>(L, idx, ObjectType::MODULE);
}

static int w_module_forward(lua_State* L) {
    ModuleWrapper* self = luax_checkmodule(L, 1);
    int numArgs = lua_gettop(L);
    std::vector<c10::IValue> values;
    for (int i = 2; i <= numArgs; i++) {
        TensorWrapper* tw = luax_checktype<TensorWrapper>(L, i, ObjectType::TENSOR);
        values.push_back(tw->getTensor());
    }
    TensorWrapper* newTensor;
    luax_catchexcept(L, [&]() { newTensor = self->forward(values); });
    if (newTensor != nullptr) {
        luax_pushtype(L, newTensor);
        newTensor->release();
    }

    return 1;
}

static int w_module_cuda(lua_State* L) {
    ModuleWrapper* self = luax_checkmodule(L, 1);
    self->cuda();

    return 1;
}

static int w_module_cpu(lua_State* L) {
    ModuleWrapper* self = luax_checkmodule(L, 1);
    self->cpu();

    return 1;
}

static int w_module_toDevice(lua_State* L) {
    ModuleWrapper* self = luax_checkmodule(L, 1);
    const char* deviceIdentifier = luaL_checkstring(L, 2);
    self->toDevice(deviceIdentifier);

    return 1;
}

static int w_module_zeroGrad(lua_State* L) {
    ModuleWrapper* self = luax_checkmodule(L, 1);
    luax_catchexcept(L, [&]() { self->zeroGrad(); }); 

    return 1;
}

static int w_module_step(lua_State* L) {
    ModuleWrapper* self = luax_checkmodule(L, 1);
    double learningRate = luaL_checknumber(L, 2);
    self->step(learningRate);

    return 1;
}

static int w_module_save(lua_State* L) {
    ModuleWrapper* self = luax_checkmodule(L, 1);
    const char* path = luaL_checkstring(L, 2);
    self->save(path);

    return 1;
}

struct luaL_Reg module_functions[] = {
    { "cuda", w_module_cuda },
    { "cpu", w_module_cpu },
    { "toDevice", w_module_toDevice },
    { "forward", w_module_forward },
    { "zeroGrad", w_module_zeroGrad },
    { "step", w_module_step },
    { "save", w_module_save },
    { "__call", w_module_forward },
    { 0, 0 }
};

int w_newModule(lua_State* L) {
    const char* path = luaL_checkstring(L, 1);
    ModuleWrapper* wrapper;
    luax_catchexcept(L, [&]() { wrapper = new ModuleWrapper(path); });
    luax_pushtype(L, wrapper);
    wrapper->release();

    return 1;
}

static int w_newTensor(lua_State* L) {
    if (lua_istable(L, 1)) {
        if (lua_gettop(L) > 1) {
            return luaL_argerror(L, 2, "expected just 1 argument, got 2 or more arguments");
        }
        size_t len = lua_objlen(L, 1);
        float* values = new float[len];
        for (size_t i = 1; i <= len; i++) {
            lua_pushnumber(L, (double) i);
            lua_gettable(L, 1);
            double value = luaL_checknumber(L, -1);
            lua_pop(L, 1);
            values[i - 1] = (float) value;
        }
        TensorWrapper* tensorWrapper = new TensorWrapper(values, len);
        luax_pushtype(L, tensorWrapper);
        tensorWrapper->release();
    } else {
        void* data = lua_touserdata(L, 1);
        lua_Number numBytes = luaL_checknumber(L, 2);
        lua_Number width = luaL_checknumber(L, 3);
        lua_Number height = luaL_checknumber(L, 4);
        const char* format = luaL_checkstring(L, 5);
        TensorWrapper* tensorWrapper;
        luax_catchexcept(L, [&](){ tensorWrapper = new TensorWrapper(data, (size_t) numBytes, (int)width, (int)height, format); });
        luax_pushtype(L, tensorWrapper);
        tensorWrapper->release();
    }

    return 1;
}

struct luaL_Reg lamp_functions[] = {
    { "newModule", w_newModule },
    { "newTensor", w_newTensor },
    { 0, 0 }
};

extern "C" __declspec(dllexport) int luaopen_lamp(lua_State * L) {
    luaL_register(L, "lamp", lamp_functions);
    luax_registertype(L, ModuleWrapper::name, module_functions);
    luax_registertype(L, TensorWrapper::name, tensor_functions);

    return 1;
}
