# Every variable in subdir must be prefixed with subdir (emulating a namespace)
TT_METAL_IMPL_LIB = $(LIBDIR)/libtt_metal_impl.a
TT_METAL_IMPL_DEFINES = -DGIT_HASH=$(shell git rev-parse HEAD)
TT_METAL_IMPL_INCLUDES = $(COMMON_INCLUDES) -I$(TT_METAL_HOME)/tt_metal/impl -I$(TT_METAL_HOME)/.
TT_METAL_IMPL_LDFLAGS = -L$(TT_METAL_HOME) -lcommon -lllrt
TT_METAL_IMPL_CFLAGS = $(CFLAGS) -Werror -Wno-int-to-pointer-cast

TT_METAL_IMPL_SRCS = \
	tt_metal/impl/device/device.cpp \
	tt_metal/impl/buffers/buffer.cpp \
	tt_metal/impl/buffers/interleaved_buffer.cpp \
	tt_metal/impl/kernels/kernel_args.cpp \
	tt_metal/impl/kernels/kernel.cpp \
	tt_metal/impl/memory_manager/allocators/free_list.cpp \
	tt_metal/impl/memory_manager/memory_manager.cpp \
	tt_metal/impl/program.cpp \

TT_METAL_IMPL_OBJS = $(addprefix $(OBJDIR)/, $(TT_METAL_IMPL_SRCS:.cpp=.o))
TT_METAL_IMPL_DEPS = $(addprefix $(OBJDIR)/, $(TT_METAL_IMPL_SRCS:.cpp=.d))

-include $(TT_METAL_IMPL_DEPS)

# Each module has a top level target as the entrypoint which must match the subdir name
tt_metal/impl: $(TT_METAL_IMPL_LIB)

$(TT_METAL_IMPL_LIB): $(COMMON_LIB) $(TT_METAL_IMPL_OBJS)
	@mkdir -p $(@D)
	ar rcs -o $@ $(TT_METAL_IMPL_OBJS)

$(OBJDIR)/tt_metal/impl/%.o: tt_metal/impl/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(TT_METAL_IMPL_CFLAGS) $(CXXFLAGS) $(STATIC_LIB_FLAGS) $(TT_METAL_IMPL_INCLUDES) $(TT_METAL_IMPL_DEFINES) -c -o $@ $<
