
set(VECADD_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/vecadd/vecadd.cpp)
set(MATMUL_SINGLE_CORE_DECODE_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/matmul_single_core_decode/matmul_single_core_decode.cpp)
set(HISTOGRAM1D_SINGLE_CORE_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/histogram1d_single_core/histogram1d_single_core.cpp)

CREATE_PGM_EXAMPLES_EXE("${VECADD_SRCS}" "contributed") # output binaries to build/programming_examples/contributed
CREATE_PGM_EXAMPLES_EXE("${MATMUL_SINGLE_CORE_DECODE_SRCS}" "contributed") # output binaries to build/programming_examples/contributed

add_custom_target(contributed DEPENDS ${PROGRAMMING_EXAMPLES_TEST_TARGETS})
