#include "CameraModel/ProjCameraModel.hpp"
#include "CameraModel/OmniCameraModel.hpp"

#include "Util/globalFuncs.hpp"

using namespace lsd_slam;

void testProjectPoint(CameraModel *omni, CameraModel *proj, vec3 point) {
	vec2 pixOmni = omni->camToPixel(point);
	vec2 pixProj = proj->camToPixel(point);
	std::cout << "Input direction: \n" << point <<
		"\nProjection Omni:\n" << pixOmni <<
		"\nProjection Proj:\n" << pixProj << std::endl;
}

void testProjectPixel(CameraModel *omni, CameraModel *proj, vec2 pixel) {
	vec3 pointOmni = omni->pixelToCam(pixel).normalized();
	vec3 pointProj = proj->pixelToCam(pixel).normalized();
	std::cout << "Input pixel: \n" << pixel <<
		"\nInv Projection Omni:\n" << pointOmni <<
		"\nInv Projection Proj:\n" << pointProj << std::endl;
}

int main(int argc, char **argv) {
	std::unique_ptr<CameraModel> model = CameraModel::loadFromFile(resourcesDir() + argv[1]);
	std::unique_ptr<CameraModel> pModel = model->makeProjCamModel();

	std::cout << "Loaded camera model from file: \"" << argv[1] << "\"" << std::endl;

	testProjectPoint(model.get(), pModel.get(), vec3(0.f, 0.f, 1.f));
	testProjectPoint(model.get(), pModel.get(), vec3(0.2f, 0.f, 1.f));
	testProjectPoint(model.get(), pModel.get(), vec3(0.2f, 0.2f, 1.f));

	testProjectPixel(model.get(), pModel.get(), vec2(200.f, 200.f));
	testProjectPixel(model.get(), pModel.get(), vec2(300.f, 300.f));

	vec2 fovAnglesOmni = model->getFovAngles();
	vec2 fovAnglesProj = pModel->getFovAngles();

	std::cout << "FOV angle Omni:\n" << fovAnglesOmni << "\n" << std::endl;
	std::cout << "FOV angle Proj:\n" << fovAnglesProj << "\n" << std::endl;

	return 0;
}
