cmake_minimum_required(VERSION 3.25.2)
project(srlts LANGUAGES CXX)

find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(LevinTree LevinTree.cpp)
