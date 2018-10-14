set(SOURCE_FILES
	"CameraControls.h"
	"CameraControls.cpp"
    "rasterize.cu"
    "rasterize.h"
    "rasterizeTools.h"
    )

cuda_add_library(src
    ${SOURCE_FILES}
    OPTIONS -arch=sm_60
    )
