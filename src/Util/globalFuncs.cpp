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

#include "Util/globalFuncs.hpp"
#include "Util/SophusUtil.hpp"
#include "opencv2/opencv.hpp"
#include "DataStructures/Frame.hpp"
#include <boost/filesystem.hpp>

namespace lsd_slam
{


SE3 SE3CV2Sophus(const cv::Mat &R, const cv::Mat &t)
{
	Sophus::Matrix3f sR;
	Sophus::Vector3f st;

	for(int i=0;i<3;i++)
	{
		sR(0,i) = static_cast<float>(R.at<double>(0,i));
		sR(1,i) = static_cast<float>(R.at<double>(1,i));
		sR(2,i) = static_cast<float>(R.at<double>(2,i));
		st[i] = static_cast<float>(t.at<double>(i));
	}

	return SE3(toSophus(sR.inverse()), toSophus(st));
}

void printMessageOnCVImage(cv::Mat &image, std::string line1,std::string line2)
{
	for(int x=0;x<image.cols;x++)
		for(int y=image.rows-30; y<image.rows;y++)
			image.at<cv::Vec3b>(y,x) *= 0.5;

	cv::putText(image, line2, cvPoint(10,image.rows-5),
	    CV_FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(200,200,250), 1, 8);

	cv::putText(image, line1, cvPoint(10,image.rows-18),
	    CV_FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(200,200,250), 1, 8);
}


cv::Mat getDepthRainbowPlot(Frame* kf, int lvl)
{
	return getDepthRainbowPlot(kf->idepth(lvl), kf->idepthVar(lvl), kf->image(lvl),
			kf->width(lvl), kf->height(lvl));
}

cv::Mat getDepthRainbowPlot(const float* idepth, const float* idepthVar, const float* gray, int width, int height)
{
	cv::Mat res = cv::Mat(height,width,CV_8UC3);
	if(gray != 0)
	{
		cv::Mat keyframeImage(height, width, CV_32F, const_cast<float*>(gray));
		cv::Mat keyframeImage8u;
		keyframeImage.convertTo(keyframeImage8u, CV_8UC1);
		cv::cvtColor(keyframeImage8u, res, CV_GRAY2RGB);
	}
	else
		fillCvMat(&res,cv::Vec3b(255,170,168));

	for(int i=0;i<width;i++)
		for(int j=0;j<height;j++)
		{
			float id = idepth[i + j*width];

			if(id >=0 && idepthVar[i + j*width] >= 0)
			{

				// rainbow between 0 and 4
				float r = (0-id) * 255 / 1.0f; if(r < 0) r = -r;
				float g = (1-id) * 255 / 1.0f; if(g < 0) g = -g;
				float b = (2-id) * 255 / 1.0f; if(b < 0) b = -b;

				uchar rc = uchar(r < 0 ? 0 : (r > 255 ? 255 : r));
				uchar gc = uchar(g < 0 ? 0 : (g > 255 ? 255 : g));
				uchar bc = uchar(b < 0 ? 0 : (b > 255 ? 255 : b));

				res.at<cv::Vec3b>(j,i) = cv::Vec3b(255-rc,255-gc,255-bc);
			}
		}
	return res;
}
cv::Mat getVarRedGreenPlot(const float* idepthVar, const float* gray, int width, int height)
{
	float* idepthVarExt = (float*)Eigen::internal::aligned_malloc(width*height*sizeof(float));

	memcpy(idepthVarExt,idepthVar,sizeof(float)*width*height);

	for(int i=2;i<width-2;i++)
		for(int j=2;j<height-2;j++)
		{
			if(idepthVar[(i) + width*(j)] <= 0)
				idepthVarExt[(i) + width*(j)] = -1;
			else
			{
				float sumIvar = 0;
				float numIvar = 0;
				for(int dx=-2; dx <=2; dx++)
					for(int dy=-2; dy <=2; dy++)
					{
						if(idepthVar[(i+dx) + width*(j+dy)] > 0)
						{
							float distFac = (float)(dx*dx+dy*dy)*(0.075f*0.075f)*0.02f;
							float ivar = 1.0f/(idepthVar[(i+dx) + width*(j+dy)] + distFac);
							sumIvar += ivar;
							numIvar += 1;
						}
					}
				idepthVarExt[(i) + width*(j)] = numIvar / sumIvar;
			}

		}


	cv::Mat res = cv::Mat(height,width,CV_8UC3);
	if(gray != 0)
	{
		cv::Mat keyframeImage(height, width, CV_32F, const_cast<float*>(gray));
		cv::Mat keyframeImage8u;
		keyframeImage.convertTo(keyframeImage8u, CV_8UC1);
		cv::cvtColor(keyframeImage8u, res, CV_GRAY2RGB);
	}
	else
		fillCvMat(&res,cv::Vec3b(255,170,168));

	for(int i=0;i<width;i++)
		for(int j=0;j<height;j++)
		{
			float idv = idepthVarExt[i + j*width];

			if(idv > 0)
			{
				float var= sqrt(idv);

				var = var*60*255*0.5f - 20;
				if(var > 255) var = 255;
				if(var < 0) var = 0;

				res.at<cv::Vec3b>(j,i) = cv::Vec3b(0,uchar(255-var), uchar(var));
			}
		}

	Eigen::internal::aligned_free((void*)idepthVarExt);

	return res;
}

void processImageWithProgressBar(cv::Size size, std::function<void(int, int)> procPixel)
{
	size_t currPercent = 0;
	for (size_t r = 0; r < size_t(size.height); ++r) {
		size_t percent = size_t((float(r) / float(size.height)) * 100.f);
		if (percent > currPercent) {
			std::cout << "\b\b\b\b" << percent << "%";
			std::cout.flush();
			percent = currPercent;
		}
		for (size_t c = 0; c < size_t(size.width); ++c) {
			procPixel(r, c);
		}
	}
	std::cout << "\b\b\b\b100%" << std::endl;
}

///\brief Ensure that an empty directory of the given name exists.
///\note If the directory exists, its contents will be deleted.
///\note Fails if the path contains more than one directory which does
///      not yet exist.
///\note Fails if the path exists and is a file, not a directory.
void makeEmptyDirectory(const std::string &dirPath)
{
	boost::filesystem::path path(dirPath);
	if (!boost::filesystem::exists(path)) {
		//Folder doesn't exist yet: just make it.
		boost::filesystem::create_directory(path);
	} else {
		if (boost::filesystem::is_regular_file(path)) {
			throw std::runtime_error("Specified path exists and is a file!");
		}
		else if (boost::filesystem::is_directory(path)) {
			//Folder exists: delete all its contents.
			for (boost::filesystem::directory_iterator end, it(path); it != end; ++it) {
				boost::filesystem::remove_all(*it);
			}
		} else {
			throw std::runtime_error("Specified path exists and is not a file or directory!");
		}
	}
}

std::ostream &operator << (std::ostream &s, const SE3 &t)
{
	s << "SE3 Element:\n\tRotation:\n" <<
		t.rotationMatrix() << "\n\tTranslation:\n" <<
		t.translation() << "\n";
	return s;
}

std::ostream &operator << (std::ostream &s, const Sim3 &t)
{

	s << "Sim3 Element:\n\tRotation:\n" <<
		t.rotationMatrix() << "\n\tTranslation:\n" <<
		t.translation() << "\nScale: " << t.scale() << "\n";
	return s;
}


}
