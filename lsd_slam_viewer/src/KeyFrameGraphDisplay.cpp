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
* along with dvo. If not, see <http://www.gnu.org/licenses/>.
*/


#ifdef _WIN32
#define NOMINMAX
#endif

#include "KeyFrameGraphDisplay.hpp"
#include "KeyFrameDisplay.hpp"
#include "settings.hpp"
#include "Util/settings.hpp"
#include <sstream>
#include <fstream>
#include <algorithm>

namespace lsd_slam {

KeyFrameGraphDisplay::KeyFrameGraphDisplay()
{
	flushPointcloud = false;
	printNumbers = false;
}

KeyFrameGraphDisplay::~KeyFrameGraphDisplay()
{
	for(unsigned int i=0;i<keyframes.size();i++)
		delete keyframes[i];
}

void KeyFrameGraphDisplay::draw()
{
	dataMutex.lock();
	numRefreshedAlready = 0;

	// draw keyframes
	float color[3] = {0,0,1};
	for(unsigned int i=0;i<keyframes.size();i++)
	{
		if(showKFCameras)
			keyframes[i]->drawCam(lineTesselation, color);

		if((showKFPointclouds && (int)i > cutFirstNKf) || i == keyframes.size()-1)
			keyframes[i]->drawPC(pointTesselation, 1);
	}


	if(flushPointcloud)
	{
		int numpts = 0;
		{
			printf("Flushing Pointcloud to %s!\n", "./pc_tmp.ply");
			std::ofstream f(resourcesDir() + "/pc_tmp.ply", std::ios::binary);
			for (unsigned int i = 0; i < keyframes.size(); i++)
			{
				if ((int)i > cutFirstNKf)
					numpts += keyframes[i]->flushPC(&f);
			}
		}

		std::ofstream f2(resourcesDir() + "/pc.ply", std::ios::binary);
		f2 << std::string("ply\n");
		f2 << std::string("format binary_little_endian 1.0\n");
		f2 << std::string("element vertex ") << numpts << std::string("\n");
		f2 << std::string("property float x\n");
		f2 << std::string("property float y\n");
		f2 << std::string("property float z\n");
		f2 << std::string("property float intensity\n");
		f2 << std::string("end_header\n");

		std::ifstream f3("./pc_tmp.ply", std::ios::binary);
		while(!f3.eof()) f2.put(f3.get());

		f2.close();
		f3.close();

		std::remove((resourcesDir() + "/pc_tmp.ply").c_str());
		flushPointcloud = false;
		printf("Done Flushing Pointcloud with %d points!\n", numpts);

	}

	if(printNumbers)
	{
		int totalPoint = 0;
		int visPoints = 0;
		for(unsigned int i=0;i<keyframes.size();i++)
		{
			totalPoint += keyframes[i]->totalPoints;
			visPoints += keyframes[i]->displayedPoints;
		}

		printf("Have %d points, %d keyframes, %d constraints. Displaying %d points.\n",
				totalPoint, (int)keyframes.size(), (int)constraints.size(), visPoints);
		printNumbers = false;
	}

	if(showConstraints)
	{
		// draw constraints
		glLineWidth(lineTesselation);
		glBegin(GL_LINES);
		for(unsigned int i=0;i<constraints.size();i++)
		{
			if(constraints[i].from == 0 || constraints[i].to == 0)
				continue;

			double m = std::min(1.0, constraints[i].err / 0.05);
			double colorScalar = std::max(0.0, m);
			glColor3f(colorScalar, 1 - colorScalar, 0);


			Sophus::Vector3f t = constraints[i].from->camToWorld.translation();
			glVertex3f((GLfloat) t[0],(GLfloat) t[1], (GLfloat) t[2]);

			t = constraints[i].to->camToWorld.translation();
			glVertex3f((GLfloat) t[0],(GLfloat) t[1], (GLfloat) t[2]);

		}
		glEnd();
	}

	dataMutex.unlock();
}

void KeyFrameGraphDisplay::addMsg(const keyframeMsg *msg)
{
	dataMutex.lock();
	if(keyframesByID.count(msg->id) == 0)
	{
		KeyFrameDisplay* disp = new KeyFrameDisplay();
		keyframesByID[msg->id] = disp;
		keyframes.push_back(disp);

	//	printf("added new KF, now there are %d!\n", (int)keyframes.size());
	}

	keyframesByID[msg->id]->setFrom(msg);
	dataMutex.unlock();
}

void KeyFrameGraphDisplay::addGraphMsg(const keyframeGraphMsg *msg)
{
	dataMutex.lock();

	constraints.resize(msg->numConstraints);
	assert(msg->constraintsData.size() == sizeof(GraphConstraint)*msg->numConstraints);
	GraphConstraint* constraintsIn = (GraphConstraint*)msg->constraintsData.data();
	for(unsigned int i=0;i<msg->numConstraints;i++)
	{
		constraints[i].err = constraintsIn[i].err;
		constraints[i].from = 0;
		constraints[i].to = 0;

		if(keyframesByID.count(constraintsIn[i].from) != 0)
			constraints[i].from = keyframesByID[constraintsIn[i].from];
//		else
//			printf("ERROR: graph update contains constraints for %d -> %d, but I dont have a frame %d!\n",
//					constraintsIn[i].from,
//					constraintsIn[i].to,
//					constraintsIn[i].from);


		if(keyframesByID.count(constraintsIn[i].to) != 0)
			constraints[i].to = keyframesByID[constraintsIn[i].to];
//		else
//			printf("ERROR: graph update contains constraints for %d -> %d, but I dont have a frame %d!\n",
//					constraintsIn[i].from,
//					constraintsIn[i].to,
//					constraintsIn[i].to);
	}



	GraphFramePose* graphPoses = (GraphFramePose*)msg->frameData.data();
	int numGraphPoses = msg->numFrames;
	assert(msg->frameData.size() == sizeof(GraphFramePose)*msg->numFrames);

	for(int i=0;i<numGraphPoses;i++)
	{
		if(keyframesByID.count(graphPoses[i].id) == 0)
		{
		//	printf("ERROR: graph update contains pose for frame %d, but I dont have a frame %d!\n", graphPoses[i].id, graphPoses[i].id);
		}
		else
			memcpy(keyframesByID[graphPoses[i].id]->camToWorld.data(), graphPoses[i].camToWorld, 7*sizeof(float));
	}

	dataMutex.unlock();

//	printf("graph update: %d constraints, %d poses\n", msg->numConstraints, msg->numFrames);
}

}

