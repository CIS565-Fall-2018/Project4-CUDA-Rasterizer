set(SOURCE_FILES
    "rasterize.cu"
    "rasterize.h"
    "rasterizeTools.h"
    "common.h"
    "common.cu"
    )

cuda_add_library(src
    ${SOURCE_FILES}
    OPTIONS -arch=sm_60
    )
