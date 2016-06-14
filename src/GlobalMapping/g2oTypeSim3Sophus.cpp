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

#include "GlobalMapping/g2oTypeSim3Sophus.hpp"

#include <g2o/core/factory.h>
#include <g2o/stuff/macros.h>

namespace lsd_slam
{


G2O_USE_TYPE_GROUP(sba);

G2O_REGISTER_TYPE_GROUP(sim3sophus);

G2O_REGISTER_TYPE(VERTEX_SIM3_SOPHUS:EXPMAP, VertexSim3);
G2O_REGISTER_TYPE(EDGE_SIM3_SOPHUS:EXPMAP, EdgeSim3);

VertexSim3::VertexSim3() : g2o::BaseVertex<7, Sophus::Sim3d>()
{
	_marginalized=false;
	_fix_scale = false;
}

bool VertexSim3::write(std::ostream& os) const
{
	assert(false);
	return false;
}

bool VertexSim3::read(std::istream& is)
{
	assert(false);
	return false;
}


EdgeSim3::EdgeSim3() :
	g2o::BaseBinaryEdge<7, Sophus::Sim3d, VertexSim3, VertexSim3>()
{
}

bool EdgeSim3::write(std::ostream& os) const
{
	assert(false);
	return false;
}

bool EdgeSim3::read(std::istream& is)
{
	assert(false);
	return false;
}
}
