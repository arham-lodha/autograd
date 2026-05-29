cmake_minimum_required(VERSION 3.20)
project(autograd CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ── Eigen ──────────────────────────────────────────────────────────────────────
include(FetchContent)
FetchContent_Declare(
  eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG        3.4.0
  GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(eigen)

# ── Catch2 (tests) ─────────────────────────────────────────────────────────────
FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.7.1
  GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(Catch2)

# ── Library ────────────────────────────────────────────────────────────────────
add_library(autograd
  src/symbol.cpp
  src/graph.cpp
  src/compiler.cpp
  src/executor.cpp
)
target_include_directories(autograd PUBLIC include)
target_link_libraries(autograd PUBLIC Eigen3::Eigen)
target_compile_options(autograd PRIVATE -Wall -Wextra -Wpedantic)

# ── Examples ───────────────────────────────────────────────────────────────────
add_executable(basic examples/basic.cpp)
target_link_libraries(basic PRIVATE autograd)

# ── Tests ──────────────────────────────────────────────────────────────────────
enable_testing()
add_subdirectory(tests)
