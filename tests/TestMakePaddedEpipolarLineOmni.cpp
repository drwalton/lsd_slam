#include "DepthEstimation/DepthMapOmniStereo.hpp"
#include <opencv2/opencv.hpp>
#include "Util/globalFuncs.hpp"
#include "Util/ModelLoader.hpp"
#include "Util/Raycast.hpp"

using namespace lsd_slam;

OmniCameraModel *omCamModel;

void showResult(MakePaddedLineErrorCode code,
	const LineSeg3d &refframeLine, const LineSeg3d &keyframeLine)
{
	if(code == MakePaddedLineErrorCode::SUCCESS) {
		std::cout << "Successfully padded line!\nIn ref frame:\n"
			<< refframeLine.start << "\n->\n" << refframeLine.end <<
			"\nIn keyframe:\n" << keyframeLine.start << "\n->\n" <<
			keyframeLine.end << std::endl;
	}
	else if(code == MakePaddedLineErrorCode::FAIL_NEAR_EPIPOLE) {
		std::cout << "Failed to pad line: too near epipole!" << std::endl;
	}
	else if(code == MakePaddedLineErrorCode::FAIL_TOO_LONG) {
		std::cout << "Failed to pad line: line too long!" << std::endl;
	}
	else if(code == MakePaddedLineErrorCode::FAIL_OUT_OF_IMAGE) {
		std::cout << "Failed to pad line: line out of image!" << std::endl;
	}
}

int main(int argc, char **argv)
{
	if(argc < 2 || argv[1] == std::string("-h")) {
		std::cout << "TestTraceEpipolarLine: test tracing lengths of an epipolar line\n"
			"\nUsage: ./TestEpipolarLine [camModel]" << std::endl;
		return 0;
	}
	
	std::unique_ptr<CameraModel> camModel = CameraModel::loadFromFile(resourcesDir() + argv[1]);
	omCamModel = dynamic_cast<OmniCameraModel*>(camModel.get());
	if(!omCamModel) {
		std::cout << "Loaded camera model not omnidirectional!" << std::endl;
		return -2;
	}
	RigidTransform keyframeToReference;
	keyframeToReference.translation.y() -= 1.f;
	float eIDepth = 0.96f;
	float maxIDepth = 0.98f;
	float minIDepth = 0.92f;
	LineSeg3d keyframeLine, refframeLine;
	vec3 keyframeDir;
	OmniEpLine2d refframeLinePix;

	std::cout << "*** Trying line 1: Straight forwards ***" << std::endl;
	MakePaddedLineErrorCode code = makePaddedEpipolarLineOmni(200, 200, eIDepth,
		minIDepth, maxIDepth, 5.f,
		*omCamModel, keyframeToReference, &keyframeDir,
		&keyframeLine, &refframeLine,
		&refframeLinePix);
	showResult(code, refframeLine, keyframeLine);

	std::cout << "*** Trying line 2: Straight down ***" << std::endl;
	code = makePaddedEpipolarLineOmni(200, 400, eIDepth,
		minIDepth, maxIDepth, 5.f,
		*omCamModel, keyframeToReference, &keyframeDir,
		&keyframeLine, &refframeLine,
		&refframeLinePix);
	showResult(code, refframeLine, keyframeLine);

	std::cout << "*** Trying line 3: Nearly straight down ***" << std::endl;
	code = makePaddedEpipolarLineOmni(200, 324, eIDepth,
		minIDepth, maxIDepth, 5.f,
		*omCamModel, keyframeToReference, &keyframeDir,
		&keyframeLine, &refframeLine,
		&refframeLinePix);
	showResult(code, refframeLine, keyframeLine);
	return 0;
}
