#include "Util/ModelLoader.hpp"
#include "ViewerOutput3DWrapper.hpp"
#include "PointCloudViewer.hpp"
#include "DataStructures/Frame.hpp"
#include "KeyFrameDisplay.hpp"
#include "GlobalMapping/KeyFrameGraph.hpp"
#include "KeyFrameGraphDisplay.hpp"
#include <qapplication.h>

struct Position {
	float x, y, z;
};

struct Orientation {
	float x, y, z, w;
};

struct Pose {
	Position position;
	Orientation orientation;
};

struct Header {
	double stamp;
	std::string frame_id;
};

struct PoseStamped {
	Pose pose;
	Header header;
};

namespace lsd_slam {

ViewerOutput3DWrapper::ViewerOutput3DWrapper(bool showViewer, int width, int height)
	:publishLevel_(0), viewer_(nullptr),
	saveKeyframeCloudsToDisk_(false)
{
	if (showViewer) {
		viewer_.reset(new PointCloudViewer());
		viewer_->show();
#ifdef _WIN32
		if (glewInit() != GLEW_OK) {
			throw std::runtime_error("GLEW INIT FAILED");
		}
#endif
	}
}

ViewerOutput3DWrapper::~ViewerOutput3DWrapper()
{

}

void ViewerOutput3DWrapper::publishKeyframeGraph(KeyFrameGraph* graph)
{
	keyframeGraphMsg gMsg;

	graph->edgesListsMutex.lock();
	gMsg.numConstraints = graph->edgesAll.size();
	gMsg.constraintsData.resize(gMsg.numConstraints * sizeof(GraphConstraint));
	GraphConstraint* constraintData = (GraphConstraint*)gMsg.constraintsData.data();
	for (unsigned int i = 0; i<graph->edgesAll.size(); i++)
	{
		constraintData[i].from = graph->edgesAll[i]->firstFrame->id();
		constraintData[i].to = graph->edgesAll[i]->secondFrame->id();
		Sophus::Vector7d err = graph->edgesAll[i]->edge->error();
		constraintData[i].err = sqrt(err.dot(err));
	}
	graph->edgesListsMutex.unlock();

	graph->keyframesAllMutex.lock_shared();
	gMsg.numFrames = graph->keyframesAll.size();
	gMsg.frameData.resize(gMsg.numFrames * sizeof(GraphFramePose));
	GraphFramePose* framePoseData = (GraphFramePose*)gMsg.frameData.data();
	for (unsigned int i = 0; i<graph->keyframesAll.size(); i++)
	{
		framePoseData[i].id = graph->keyframesAll[i]->id();
		memcpy(framePoseData[i].camToWorld, graph->keyframesAll[i]->getScaledCamToWorld().cast<float>().data(), sizeof(float) * 7);
	}
	graph->keyframesAllMutex.unlock_shared();

	if(viewer_) viewer_->addGraphMsg(&gMsg);
}

// publishes a keyframe. if that frame already existis, it is overwritten, otherwise it is added.
void ViewerOutput3DWrapper::publishKeyframe(Frame* kf)
{
	keyframeMsg msg;
	msg.id = kf->id();
	msg.time = kf->timestamp();
	msg.isKeyframe = true;

	size_t w = kf->width(), h = kf->height();
	msg.width = w;
	msg.height = h;
	msg.modelType = kf->model().getType();
	msg.fx = kf->model(publishLevel_).fx;
	msg.fy = kf->model(publishLevel_).fy;
	msg.cx = kf->model(publishLevel_).cx;
	msg.cy = kf->model(publishLevel_).cy;
	if (msg.modelType == CameraModelType::OMNI) {
		msg.e = static_cast<const OmniCameraModel&>(kf->model(publishLevel_)).e;
	}
	auto ctw = kf->getScaledCamToWorld().cast<float>();
	memcpy(&(msg.camToWorld), ctw.data(), 7 * sizeof(float));

	msg.pointcloud.resize(w*h*sizeof(InputPointDense));
	InputPointDense *ip = reinterpret_cast<InputPointDense*>(msg.pointcloud.data());

	const float *idepth = kf->idepth(publishLevel_);
	const float *idepthVar = kf->idepthVar(publishLevel_);
	const float *color = kf->image(publishLevel_);

	for (size_t i = 0; i < w*h; ++i) {
		ip[i].idepth = idepth[i];
		ip[i].idepth_var = idepthVar[i];
		ip[i].color[0] = ip[i].color[1] =
			ip[i].color[2] = ip[i].color[3] = color[i];
	}
	
	if(viewer_) viewer_->addFrameMsg(&msg);
	if (saveKeyframeCloudsToDisk_) {
		saveKeyframeCloud(kf);
	}
}

// published a tracked frame that did not become a keyframe (yet; i.e. has no depth data)
void ViewerOutput3DWrapper::publishTrackedFrame(Frame* kf)
{
	keyframeMsg msg;


	msg.id = kf->id();
	msg.time = kf->timestamp();
	msg.isKeyframe = false;


	memcpy(msg.camToWorld.data(), kf->getScaledCamToWorld().cast<float>().data(), sizeof(float) * 7);
	msg.modelType = kf->model().getType();
	msg.fx = kf->model(publishLevel_).fx;
	msg.fy = kf->model(publishLevel_).fy;
	msg.cx = kf->model(publishLevel_).cx;
	msg.cy = kf->model(publishLevel_).cy;
	if (msg.modelType == CameraModelType::OMNI) {
		msg.e = static_cast<const OmniCameraModel&>(kf->model(publishLevel_)).e;
	}
	msg.width = kf->width(publishLevel_);
	msg.height = kf->height(publishLevel_);

	msg.pointcloud.clear();

	if (viewer_) viewer_->addFrameMsg(&msg);

	SE3 camToWorld = se3FromSim3(kf->getScaledCamToWorld());

	PoseStamped pMsg;

	pMsg.pose.position.x = camToWorld.translation()[0];
	pMsg.pose.position.y = camToWorld.translation()[1];
	pMsg.pose.position.z = camToWorld.translation()[2];
	pMsg.pose.orientation.x = camToWorld.so3().unit_quaternion().x();
	pMsg.pose.orientation.y = camToWorld.so3().unit_quaternion().y();
	pMsg.pose.orientation.z = camToWorld.so3().unit_quaternion().z();
	pMsg.pose.orientation.w = camToWorld.so3().unit_quaternion().w();

	if (pMsg.pose.orientation.w < 0)
	{
		pMsg.pose.orientation.x *= -1;
		pMsg.pose.orientation.y *= -1;
		pMsg.pose.orientation.z *= -1;
		pMsg.pose.orientation.w *= -1;
	}

	pMsg.header.stamp = kf->timestamp();
	pMsg.header.frame_id = "world";

}

// publishes graph and all constraints, as well as updated KF poses.
void ViewerOutput3DWrapper::publishTrajectory(std::vector<Eigen::Matrix<float, 3, 1>> trajectory, std::string identifier)
{
	//Leaving for now.
}

void ViewerOutput3DWrapper::publishTrajectoryIncrement(Eigen::Matrix<float, 3, 1> pt, std::string identifier)
{
	//Leaving for now.
}

void ViewerOutput3DWrapper::publishDebugInfo(const Eigen::Matrix<float, 20, 1> &data)
{

}

void ViewerOutput3DWrapper::saveKeyframeCloudsToDisk(bool b)
{
	saveKeyframeCloudsToDisk_ = b;
}
bool ViewerOutput3DWrapper::saveKeyframeCloudsToDisk() const
{
	return saveKeyframeCloudsToDisk_;
}

void ViewerOutput3DWrapper::saveKeyframeCloud(const Frame *kf) const
{
	static size_t kfNo = 0;
	ModelLoader loader;
	for (size_t r = 0; r < size_t(kf->height()); ++r) {
		for (size_t c = 0; c < size_t(kf->width()); ++c) {
			if (kf->model().pixelLocValid(vec2(c, r))) {
				float depth = 1.f / const_cast<Frame*>(kf)->idepth(0)[r*kf->width() + c];
				if (depth > 0.f) {
					vec3 pt = kf->model().pixelToCam(vec2(c, r), depth);
					if (pt == pt) {
						loader.vertices().push_back(pt);
						loader.vertColors().push_back(
							const_cast<Frame*>(kf)->image(0)[r*kf->width() + c] *
							vec3(1.f, 1.f, 1.f));
					}
				}
			}
		}
	}

	std::string filename = "Keyframe_" + std::to_string(kfNo) + ".ply";
	loader.saveFile(resourcesDir() + filename);
	std::cout << "Saved keyframe to pointcloud: " << filename << std::endl;
	++kfNo;
}

}