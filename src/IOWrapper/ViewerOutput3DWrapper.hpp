/**
* This file is part of LSD-SLAM.
*
* Copyright 2013 Jakob Engel <engelj at in dot tum dot de> (Technical University of Munich)
* For more information see <http://vision.in.tum.de/lsdslam> 
*
* LSD-SLAM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* LSD-SLAM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with LSD-SLAM. If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once
#include "System/Win32Compatibility.hpp"
#include "Output3DWrapper.hpp"
#include <thread>
#include <memory>
#include <atomic>
#include "Viewer/PointCloudViewer.hpp"

namespace lsd_slam {

///\brief This implementation of Output3DWrapper generates a 3D point cloud from keyframes, and
///       optionally displays the output in a viewer window.
class ViewerOutput3DWrapper : public Output3DWrapper
{
public:
	ViewerOutput3DWrapper(bool showViewer, int width, int height);
	virtual ~ViewerOutput3DWrapper();

	virtual void publishKeyframeGraph(KeyframeGraph* graph);

	// publishes a keyframe. if that frame already existis, it is overwritten, otherwise it is added.
	virtual void publishKeyframe(Frame* kf);

	// published a tracked frame that did not become a keyframe (yet; i.e. has no depth data)
	virtual void publishTrackedFrame(Frame* kf);

	// publishes graph and all constraints, as well as updated KF poses.
	virtual void publishTrajectory(std::vector<Eigen::Matrix<float, 3, 1>> trajectory, std::string identifier);
	virtual void publishTrajectoryIncrement(Eigen::Matrix<float, 3, 1> pt, std::string identifier);

	virtual void publishDebugInfo(const Eigen::Matrix<float, 20, 1> &data);
private:
	std::thread viewerThread_;
	size_t publishLevel_;
	std::unique_ptr<PointCloudViewer> viewer_;
};

}
