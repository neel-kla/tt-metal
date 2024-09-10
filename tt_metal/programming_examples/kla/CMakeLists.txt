
set(HISTOGRAM1D_SINGLE_CORE_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/histogram1d_single_core/histogram1d_single_core.cpp)

# output binaries to build/programming_examples/kla
CREATE_PGM_EXAMPLES_EXE("${HISTOGRAM1D_SINGLE_CORE_SRCS}" "kla")
 
add_custom_target(kla DEPENDS ${PROGRAMMING_EXAMPLES_TEST_TARGETS})

