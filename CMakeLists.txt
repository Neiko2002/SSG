cmake_minimum_required(VERSION 3.15)

project(ssg)

# Boost
find_package(Boost REQUIRED)

# OpenMP
find_package(OpenMP REQUIRED COMPONENTS CXX)

# Python binding
find_package(pybind11)

# Source codes
add_subdirectory(src)
add_subdirectory(tests)
