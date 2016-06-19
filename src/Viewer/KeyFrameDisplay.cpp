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

#define GL_GLEXT_PROTOTYPES 1

#include "KeyframeDisplay.hpp"
#include "CameraModel/CameraModel.hpp"
#include <stdio.h>
#include "settings.hpp"

#ifdef __APPLE__
#include <OpenGL/OpenGL.h>
#include <OpenGL/glu.h>
#elif defined _WIN32
#include <GL/glew.h>
#else
#include <GL/gl.h>
#include <GL/glu.h>
#endif

#include "opencv2/opencv.hpp"

namespace lsd_slam
{

KeyframeDisplay::KeyframeDisplay()
{
	originalInput = 0;
	id = 0;
	vertexBufferIdValid = false;
	glBuffersValid = false;


	camToWorld = Sophus::Sim3f();

	my_scaledTH = my_absTH = 0;

	totalPoints = displayedPoints = 0;
}


KeyframeDisplay::~KeyframeDisplay()
{
	if(vertexBufferIdValid)
	{
		glDeleteBuffers(1, &vertexBufferId);
		vertexBufferIdValid = false;
	}

	if(originalInput != 0)
		delete[] originalInput;
}


void KeyframeDisplay::setFrom(const keyframeMsg *msg)
{
	// copy over campose.
	memcpy(camToWorld.data(), msg->camToWorld.data(), 7*sizeof(float));

	camModel_ = msg->getCameraModel();
	id = msg->id;
	time = msg->time;

	if(originalInput != 0)
		delete[] originalInput;
	originalInput=0;

	if(msg->pointcloud.size() != camModel_->w*camModel_->h*sizeof(InputPointDense))
	{
		if(msg->pointcloud.size() != 0)
		{
			printf("WARNING: PC with points, but number of points not right! (is %lu, should be %lu*%dx%d=%u)\n",
					msg->pointcloud.size(), sizeof(InputPointDense), camModel_->w, camModel_->h, camModel_->w*camModel_->h*sizeof(InputPointDense));
		}
	}
	else
	{
		originalInput = new InputPointDense[camModel_->w*camModel_->h];
		memcpy(originalInput, msg->pointcloud.data(), camModel_->w*camModel_->h*sizeof(InputPointDense));
	}

	glBuffersValid = false;
}

void KeyframeDisplay::refreshPC()
{
//	minNearSupport = 9;
	bool paramsStillGood = my_scaledTH == scaledDepthVarTH &&
			my_absTH == absDepthVarTH &&
			my_scale*1.2 > camToWorld.scale() &&
			my_scale < camToWorld.scale()*1.2 &&
			my_minNearSupport == minNearSupport &&
			my_sparsifyFactor == sparsifyFactor;



	if(glBuffersValid && (paramsStillGood || numRefreshedAlready > 10)) return;
	numRefreshedAlready++;

	glBuffersValid = true;


	// delete old vertex buffer
	if(vertexBufferIdValid)
	{
		glDeleteBuffers(1, &vertexBufferId);
		vertexBufferIdValid = false;
	}



	// if there are no vertices, done!
	if(originalInput == 0)
		return;


	// make data
	MyVertex* tmpBuffer = new MyVertex[camModel_->w*camModel_->h];

	my_scaledTH =scaledDepthVarTH;
	my_absTH = absDepthVarTH;
	my_scale = camToWorld.scale();
	my_minNearSupport = minNearSupport;
	my_sparsifyFactor = sparsifyFactor;
	// data is directly in ros message, in correct format.
	vertexBufferNumPoints = 0;

	int total = 0, displayed = 0;
	for(size_t y=1;y<camModel_->h-1;y++)
		for(size_t x=1;x<camModel_->w-1;x++)
		{
			if(originalInput[x+y*camModel_->w].idepth <= 0) continue;
			total++;


			if(my_sparsifyFactor > 1 && rand()%my_sparsifyFactor != 0) continue;

			float depth = 1 / originalInput[x+y*camModel_->w].idepth;
			float depth4 = depth*depth; depth4*= depth4;


			if(originalInput[x+y*camModel_->w].idepth_var * depth4 > my_scaledTH)
				continue;

			if(originalInput[x+y*camModel_->w].idepth_var * depth4 * my_scale*my_scale > my_absTH)
				continue;

			if(my_minNearSupport > 1)
			{
				int nearSupport = 0;
				for(int dx=-1;dx<2;dx++)
					for(int dy=-1;dy<2;dy++)
					{
						int idx = x+dx+(y+dy)*camModel_->w;
						if(originalInput[idx].idepth > 0)
						{
							float diff = originalInput[idx].idepth - 1.0f / depth;
							if(diff*diff < 2*originalInput[x+y*camModel_->w].idepth_var)
								nearSupport++;
						}
					}

				if(nearSupport < my_minNearSupport)
					continue;
			}

			vec3 pt = camModel_->pixelToCam(vec2(x, y), depth);
			tmpBuffer[vertexBufferNumPoints].point[0] = pt.x();
			tmpBuffer[vertexBufferNumPoints].point[1] = pt.y();
			tmpBuffer[vertexBufferNumPoints].point[2] = pt.z();

			tmpBuffer[vertexBufferNumPoints].color[3] = 100;
			tmpBuffer[vertexBufferNumPoints].color[2] = originalInput[x+y*camModel_->w].color[0];
			tmpBuffer[vertexBufferNumPoints].color[1] = originalInput[x+y*camModel_->w].color[1];
			tmpBuffer[vertexBufferNumPoints].color[0] = originalInput[x+y*camModel_->w].color[2];

			vertexBufferNumPoints++;
			displayed++;
		}
	totalPoints = total;
	displayedPoints = displayed;

	// create new ones, static
	vertexBufferId=0;
	glGenBuffers(1, &vertexBufferId);
	glBindBuffer(GL_ARRAY_BUFFER, vertexBufferId);         // for vertex coordinates
	glBufferData(GL_ARRAY_BUFFER, sizeof(MyVertex) * vertexBufferNumPoints, tmpBuffer, GL_STATIC_DRAW);
	vertexBufferIdValid = true;

	if(!keepInMemory)
	{
		delete[] originalInput;
		originalInput = 0;
	}

	delete[] tmpBuffer;
}



void KeyframeDisplay::drawCam(float lineWidth, float* color)
{
	if(camModel_.get() == nullptr)
		return;


	glPushMatrix();

		Sophus::Matrix4f m = camToWorld.matrix();
		glMultMatrixf((GLfloat*)m.data());

		if(color == 0)
			glColor3f(1,0,0);
		else
			glColor3f(color[0],color[1],color[2]);

		glLineWidth(lineWidth);
		glBegin(GL_LINES);
		glVertex3f(0,0,0);
		glVertex3f(0.05f*(0-camModel_->cx)/camModel_->fx,0.05f*(0-camModel_->cy)/camModel_->fy,0.05f);
		glVertex3f(0,0,0);
		glVertex3f(0.05f*(0-camModel_->cx)/camModel_->fx,0.05*(camModel_->h-1-camModel_->cy)/camModel_->fy,0.05f);
		glVertex3f(0,0,0);
		glVertex3f(0.05f*(camModel_->w-1-camModel_->cx)/camModel_->fx,0.05f*(camModel_->h-1-camModel_->cy)/camModel_->fy,0.05f);
		glVertex3f(0,0,0);
		glVertex3f(0.05f*(camModel_->w-1-camModel_->cx)/camModel_->fx,0.05f*(0-camModel_->cy)/camModel_->fy,0.05f);

		glVertex3f(0.05f*(camModel_->w-1-camModel_->cx)/camModel_->fx,0.05f*(0-camModel_->cy)/camModel_->fy,0.05f);
		glVertex3f(0.05f*(camModel_->w-1-camModel_->cx)/camModel_->fx,0.05f*(camModel_->h-1-camModel_->cy)/camModel_->fy,0.05f);

		glVertex3f(0.05f*(camModel_->w-1-camModel_->cx)/camModel_->fx,0.05f*(camModel_->h-1-camModel_->cy)/camModel_->fy,0.05f);
		glVertex3f(0.05f*(0-camModel_->cx)/camModel_->fx,0.05f*(camModel_->h-1-camModel_->cy)/camModel_->fy,0.05f);

		glVertex3f(0.05f*(0-camModel_->cx)/camModel_->fx,0.05f*(camModel_->h-1-camModel_->cy)/camModel_->fy,0.05f);
		glVertex3f(0.05f*(0-camModel_->cx)/camModel_->fx,0.05f*(0-camModel_->cy)/camModel_->fy,0.05f);

		glVertex3f(0.05f*(0-camModel_->cx)/camModel_->fx,0.05f*(0-camModel_->cy)/camModel_->fy,0.05f);
		glVertex3f(0.05f*(camModel_->w-1-camModel_->cx)/camModel_->fx,0.05f*(0-camModel_->cy)/camModel_->fy,0.05f);

		glEnd();
	glPopMatrix();
}

int KeyframeDisplay::flushPC(std::ofstream* f)
{

	MyVertex* tmpBuffer = new MyVertex[camModel_->w*camModel_->h];
	int num = 0;
	for(size_t y=1;y<camModel_->h-1;y++)
		for(size_t x=1;x<camModel_->w-1;x++)
		{
			if(originalInput[x+y*camModel_->w].idepth <= 0) continue;

			if(my_sparsifyFactor > 1 && rand()%my_sparsifyFactor != 0) continue;

			float depth = 1 / originalInput[x+y*camModel_->w].idepth;
			float depth4 = depth*depth; depth4*= depth4;

			if(originalInput[x+y*camModel_->w].idepth_var * depth4 > my_scaledTH)
				continue;

			if(originalInput[x+y*camModel_->w].idepth_var * depth4 * my_scale*my_scale > my_absTH)
				continue;

			if(my_minNearSupport > 1)
			{
				int nearSupport = 0;
				for(int dx=-1;dx<2;dx++)
					for(int dy=-1;dy<2;dy++)
					{
						int idx = x+dx+(y+dy)*camModel_->w;
						if(originalInput[idx].idepth > 0)
						{
							float diff = originalInput[idx].idepth - 1.0f / depth;
							if(diff*diff < 2*originalInput[x+y*camModel_->w].idepth_var)
								nearSupport++;
						}
					}

				if(nearSupport < my_minNearSupport)
					continue;
			}


			Sophus::Vector3f pt = camToWorld * (camModel_->pixelToCam(vec2(x, y)) *depth);
			tmpBuffer[num].point[0] = pt[0];
			tmpBuffer[num].point[1] = pt[1];
			tmpBuffer[num].point[2] = pt[2];



			tmpBuffer[num].color[3] = 100;
			tmpBuffer[num].color[2] = originalInput[x+y*camModel_->w].color[0];
			tmpBuffer[num].color[1] = originalInput[x+y*camModel_->w].color[1];
			tmpBuffer[num].color[0] = originalInput[x+y*camModel_->w].color[2];

			num++;
		}




	for(int i=0;i<num;i++)
	{
		f->write((const char *)tmpBuffer[i].point,3*sizeof(float));
		float color = tmpBuffer[i].color[0] / 255.0;
		f->write((const char *)&color,sizeof(float));
	}
	//	*f << tmpBuffer[i].point[0] << " " << tmpBuffer[i].point[1] << " " << tmpBuffer[i].point[2] << " " << (tmpBuffer[i].color[0] / 255.0) << "\n";

	delete[] tmpBuffer;

	printf("Done flushing frame %d (%d points)!\n", this->id, num);
	return num;
}

void KeyframeDisplay::drawPC(float pointSize, float alpha)
{
	refreshPC();

	if(!vertexBufferIdValid)
	{
		return;
	}

	GLfloat LightColor[] = {1, 1, 1, 1};
	if(alpha < 1)
	{
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		LightColor[0] = LightColor[1] = 0;
		glEnable(GL_LIGHTING);
		glDisable(GL_LIGHT1);

		glLightfv (GL_LIGHT0, GL_AMBIENT, LightColor);
	}
	else
	{
		glDisable(GL_LIGHTING);
	}


	glPushMatrix();

		Sophus::Matrix4f m = camToWorld.matrix();
		glMultMatrixf((GLfloat*)m.data());

		glPointSize(pointSize);

		glBindBuffer(GL_ARRAY_BUFFER, vertexBufferId);

		glVertexPointer(3, GL_FLOAT, sizeof(MyVertex), 0);
		glColorPointer(4, GL_UNSIGNED_BYTE, sizeof(MyVertex), (const void*) (3*sizeof(float)));

		glEnableClientState(GL_VERTEX_ARRAY);
		glEnableClientState(GL_COLOR_ARRAY);

		glDrawArrays(GL_POINTS, 0, vertexBufferNumPoints);

		glDisableClientState(GL_COLOR_ARRAY);
		glDisableClientState(GL_VERTEX_ARRAY);

	glPopMatrix();




	if(alpha < 1)
	{
		glDisable(GL_BLEND);
		glDisable(GL_LIGHTING);
		LightColor[2] = LightColor[1] = LightColor[0] = 1;
		glLightfv (GL_LIGHT0, GL_AMBIENT_AND_DIFFUSE, LightColor);
	}
}

}

