include $(TT_METAL_HOME)/libs/dtx/module.mk
include $(TT_METAL_HOME)/libs/tt_dnn/module.mk
include $(TT_METAL_HOME)/libs/ttlib/module.mk

TT_LIBS_TO_BUILD = libs/dtx \
                   libs/tt_dnn \
                   ttlib \

libs: $(TT_LIBS_TO_BUILD)
