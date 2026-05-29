add_executable(tests
  test_symbol.cpp
  test_compiler.cpp
  test_executor.cpp
)
target_link_libraries(tests PRIVATE autograd Catch2::Catch2WithMain)

include(CTest)
include(Catch)
catch_discover_tests(tests)
