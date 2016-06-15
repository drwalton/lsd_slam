#include "CameraModel/ProjCameraModel.hpp"
#include "CameraModel/OmniCameraModel.hpp"
#include "DepthEstimation/DepthMap.hpp"

#include "Util/globalFuncs.hpp"

using namespace lsd_slam;

//

int main(int argc, char **argv) {
	std::unique_ptr<CameraModel> model = CameraModel::loadFromFile(resourcesDir() + argv[1]);
	std::unique_ptr<CameraModel> pModel = model->makeProjCamModel();

	DepthMap projDepthMap();

	return 0;
}
