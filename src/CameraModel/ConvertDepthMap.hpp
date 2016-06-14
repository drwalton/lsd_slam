#include "OmniCameraModel.hpp"
#include "ProjCameraModel.hpp"
#include <iostream>

namespace lsd_slam {

///\brief Convert a depth map from omnidirectional format (dist. from camera)
/// to projective format (z-component) 
void depthMapOmniToProj(float *input, float *output, CameraModel *model);

///\brief Convert a depth map from projective format (z-component) to
/// omnidirectional format (dist. from camera)
void depthMapProjToOmni(float *input, float *output, CameraModel *model);

}
