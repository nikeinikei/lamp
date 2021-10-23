cmake_minimum_required(VERSION 3.1)
project(lamp)

find_package(Torch REQUIRED)

if(NOT LUAJIT_PATH)
    message(FATAL_ERROR "luajit path not specific, cannot link")
endif()

add_library(lamp SHARED lib.cpp)
set_target_properties(lamp PROPERTIES PREFIX "")
target_include_directories(lamp PRIVATE ${LUAJIT_PATH}/src)
if(UNIX)
    target_link_libraries(lamp ${TORCH_LIBRARIES} ${LUAJIT_PATH}/src/libluajit.so)
else()
    target_link_libraries(lamp ${TORCH_LIBRARIES} ${LUAJIT_PATH}/src/lua51.lib)
endif()
set_property(TARGET lamp PROPERTY CXX_STANDARD 14)
