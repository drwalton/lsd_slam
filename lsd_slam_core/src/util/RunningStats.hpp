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

#include <string.h>

namespace lsd_slam
{

class RunningStats
{
public:
	int num_stereo_comparisons;
	int num_stereo_calls;
	int num_pixelInterpolations;

	int num_stereo_rescale_oob;
	int num_stereo_inf_oob;
	int num_stereo_near_oob;
	int num_stereo_invalid_unclear_winner;
	int num_stereo_invalid_atEnd;
	int num_stereo_invalid_inexistantCrossing;
	int num_stereo_invalid_twoCrossing;
	int num_stereo_invalid_noCrossing;
	int num_stereo_invalid_bigErr;
	int num_stereo_interpPre;
	int num_stereo_interpPost;
	int num_stereo_interpNone;
	int num_stereo_negative;
	int num_stereo_successfull;


	int num_observe_created;
	int num_observe_blacklisted;
	int num_observe_updated;
	int num_observe_skipped_small_epl;
	int num_observe_skipped_small_epl_grad;
	int num_observe_skipped_small_epl_angle;
	int num_observe_transit_finalizing;
	int num_observe_transit_idle_oob;
	int num_observe_transit_idle_scale_angle;
	int num_observe_trans_idle_exhausted;
	int num_observe_inconsistent_finalizing;
	int num_observe_inconsistent;
	int num_observe_notfound_finalizing2;
	int num_observe_notfound_finalizing;
	int num_observe_notfound;
	int num_observe_skip_fail;
	int num_observe_skip_oob;
	int num_observe_good;
	int num_observe_good_finalizing;
	int num_observe_state_finalizing;
	int num_observe_state_initializing;


	int num_observe_skip_alreadyGood;
	int num_observe_addSkip;



	int num_observe_no_grad_removed;
	int num_observe_no_grad_left;
	int num_observe_update_attempted;
	int num_observe_create_attempted;
	int num_observe_updated_ignored;
	int num_observe_spread_unsuccessfull;

	int num_prop_removed_out_of_bounds;
	int num_prop_removed_colorDiff;
	int num_prop_removed_validity;
	int num_prop_grad_decreased;
	int num_prop_color_decreased;
	int num_prop_attempts;
	int num_prop_occluded;
	int num_prop_created;
	int num_prop_merged;

	int num_reg_created;
	int num_reg_smeared;
	int num_reg_total;
	int num_reg_deleted_secondary;
	int num_reg_deleted_occluded;
	int num_reg_blacklisted;
	int num_reg_setBlacklisted;

	inline RunningStats()
	{
		setZero();
	}

	inline void setZero()
	{
		memset(this,0,sizeof(RunningStats));
	}

	inline void add(RunningStats* r)
	{
		int* pt = (int*)this;
		int* pt_r = (int*)r;
		for(int i=0;i<static_cast<int>(sizeof(RunningStats)/sizeof(int));i++)
			pt[i] += pt_r[i];
	}
};

extern RunningStats runningStats;

}
