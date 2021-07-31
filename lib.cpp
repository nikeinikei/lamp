#include "lua.hpp"
#include "torch/script.h"

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

class TensorWrapper : public Object {
public:
	constexpr static const char* name = "lamp.Tensor";

	TensorWrapper(void* data, int width, int height, const char* format) {
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
			tensor = torch::from_blob(data, { width, height, numComponents }, options);
		}
		else {
			throw std::exception("unsupported format");
		}
	}

	TensorWrapper(const at::Tensor& t) {
		tensor = t;
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

private:
	at::Tensor tensor;
};

static int w_tensor_item(lua_State* L) {
	TensorWrapper* self = luax_checktype<TensorWrapper>(L, 1, ObjectType::TENSOR);
	double value = self->item();
	lua_pushnumber(L, value);

	return 1;
}

struct luaL_Reg tensor_functions[] = {
	{ "item", w_tensor_item },
	{ 0, 0 }
};

class ModuleWrapper : public Object {
public:
	constexpr static const char const* name = "lamp.Module";

	ModuleWrapper(const char* path) {
		try {
			mod = torch::jit::load(path);
		}
		catch (c10::Error e) {
			throw std::exception(e.what());
		}
	}

	const char* getName() override {
		return ModuleWrapper::name;
	}

	ObjectType getType() override {
		return ObjectType::MODULE;
	}

	TensorWrapper* forward(std::vector<c10::IValue> inputs) {
		try {
			return new TensorWrapper(mod.forward(inputs).toTensor());
		}
		catch (c10::Error err) {
			throw std::exception(err.what());
		}
	}

private:
	torch::jit::script::Module mod;
};

static int w_module_forward(lua_State* L) {
	ModuleWrapper* self = luax_checktype<ModuleWrapper>(L, 1, ObjectType::MODULE);
	int numArgs = lua_gettop(L);
	std::vector<c10::IValue> values;
	for (int i = 2; i <= numArgs; i++) {
		TensorWrapper* tw = luax_checktype<TensorWrapper>(L, i, ObjectType::TENSOR);
		values.push_back(tw->getTensor());
	}
	TensorWrapper* newTensor;
	luax_catchexcept(L, [&]() { newTensor = self->forward(values); });
	luax_pushtype(L, newTensor);
	newTensor->release();

	return 1;
}

struct luaL_Reg module_functions[] = {
	{ "forward", w_module_forward },
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
	void* data = lua_touserdata(L, 1);
	lua_Number width = luaL_checknumber(L, 2);
	lua_Number height = luaL_checknumber(L, 3);
	const char* format = luaL_checkstring(L, 4);
	TensorWrapper* tensorWrapper;
	luax_catchexcept(L, [&](){ tensorWrapper = new TensorWrapper(data, (int)width, (int)height, format); });
	luax_pushtype(L, tensorWrapper);
	tensorWrapper->release();

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