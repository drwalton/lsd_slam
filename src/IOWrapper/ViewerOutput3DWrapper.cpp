#include "ViewerOutput3DWrapper.hpp"
#include "Viewer/PointCloudViewer.hpp"
#include "DataStructures/Frame.hpp"
#include "Viewer/KeyFrameDisplay.hpp"
#include "GlobalMapping/KeyFrameGraph.hpp"
#include "Viewer/KeyFrameGraphDisplay.hpp"
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
	:publishLevel_(0), viewer_(nullptr)
{
	if (showViewer) {
		viewerThread_ = std::thread([this](){
			std::cout << "Launching viewer thread...\n";
			int argc = 1; 
			char* argv = "app";
			QApplication qapp(argc, &argv);
			PointCloudViewer viewer;
			viewer_ = &viewer;
			viewer.show();
			if (glewInit() != GLEW_OK) {
				throw std::runtime_error("GLEW INIT FAILED");
			}
			qapp.exec();
			viewer_ = nullptr;
			std::cout << "Terminating viewer thread...\n";
		});
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
	msg.fx = kf->fx(publishLevel_);
	msg.fy = kf->fy(publishLevel_);
	msg.cx = kf->cx(publishLevel_);
	msg.cy = kf->cy(publishLevel_);
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
}

// published a tracked frame that did not become a keyframe (yet; i.e. has no depth data)
void ViewerOutput3DWrapper::publishTrackedFrame(Frame* kf)
{
	keyframeMsg fMsg;


	fMsg.id = kf->id();
	fMsg.time = kf->timestamp();
	fMsg.isKeyframe = false;


	memcpy(fMsg.camToWorld.data(), kf->getScaledCamToWorld().cast<float>().data(), sizeof(float) * 7);
	fMsg.fx = kf->fx(publishLevel_);
	fMsg.fy = kf->fy(publishLevel_);
	fMsg.cx = kf->cx(publishLevel_);
	fMsg.cy = kf->cy(publishLevel_);
	fMsg.width = kf->width(publishLevel_);
	fMsg.height = kf->height(publishLevel_);

	fMsg.pointcloud.clear();

	if (viewer_) viewer_->addFrameMsg(&fMsg);

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

}