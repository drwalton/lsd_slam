#include "CameraModel/ProjCameraModel.hpp"
#include "CameraModel/OmniCameraModel.hpp"
#include "DepthEstimation/DepthMap.hpp"
#include "CameraModel/ConvertDepthMap.hpp"

#include "Util/globalFuncs.hpp"

using namespace lsd_slam;

//

int main(int argc, char **argv) {
	std::unique_ptr<CameraModel> model = CameraModel::loadFromFile(resourcesDir() + argv[1]);
	std::unique_ptr<CameraModel> pModel = model->makeProjCamModel();

	DepthMap projDepthMap(*pModel);
	DepthMap omniDepthMap(*model);
	/*
	{
		float *projDepths = new float[model->w*model->h];
		for (size_t i = 0; i < model->w*model->h; ++i) {
			projDepths[i] = 1.f;
		}
		float *omniDepths = new float[model->w*model->h];
		depthMapProjToOmni(projDepths, omniDepths, model.get());

		projDepthMap.initializeFromGTDepth()
	}
	*/

	//projDepthMap.initializeRandomly();
	//omniDepthMap.initializeRandomly();

	return 0;
}
