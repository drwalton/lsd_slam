#include "ViewerOutput3DWrapper.hpp"

namespace lsd_slam {

ViewerOutput3DWrapper::ViewerOutput3DWrapper(bool showViewer, int width, int height)
{

}
ViewerOutput3DWrapper::~ViewerOutput3DWrapper()
{

}

void ViewerOutput3DWrapper::publishKeyframeGraph(KeyFrameGraph* graph)
{

}

// publishes a keyframe. if that frame already existis, it is overwritten, otherwise it is added.
void ViewerOutput3DWrapper::publishKeyframe(Frame* kf)
{

}

// published a tracked frame that did not become a keyframe (yet; i.e. has no depth data)
void ViewerOutput3DWrapper::publishTrackedFrame(Frame* kf)
{

}

// publishes graph and all constraints, as well as updated KF poses.
void ViewerOutput3DWrapper::publishTrajectory(std::vector<Eigen::Matrix<float, 3, 1>> trajectory, std::string identifier)
{
	
}
void ViewerOutput3DWrapper::publishTrajectoryIncrement(Eigen::Matrix<float, 3, 1> pt, std::string identifier)
{

}

void ViewerOutput3DWrapper::publishDebugInfo(const Eigen::Matrix<float, 20, 1> &data)
{

}

}