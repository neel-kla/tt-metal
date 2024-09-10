### Instructions to build

1. Untar the kla.tar.gz file
2. Update the programming examples CMakeLists.txt with following lines

    ` add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/kla) `
    ` add_custom_target(programming_examples DEPENDS profiler_examples contributed kla ${PROGRAMMING_EXAMPLES_TEST_TARGETS}) `
3. ./build_metal.sh to build programming examples along with contributed and kla samples that includes histogram
4. The expected output is being captured in the "kla/histogram_unary_gt_output.txt" file