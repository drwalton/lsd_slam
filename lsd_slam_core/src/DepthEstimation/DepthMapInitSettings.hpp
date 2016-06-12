#pragma once

namespace lsd_slam
{

/** ============== Depth Variance Handling ======================= */
// before an ekf-update, the variance is increased by this factor.
#define SUCC_VAR_INC_FAC (1.01f)

// after a failed stereo observation, the variance is increased by this factor.
#define FAIL_VAR_INC_FAC 1.1f

// initial variance on creation -
// if variance becomes larger than this, hypothesis is removed.
#define MAX_VAR (0.5f*0.5f)

// initial variance for Ground Truth Initialisation
#define VAR_GT_INIT_INITIAL 0.01f*0.01f

// initial variance for Random Initialisation
#define VAR_RANDOM_INIT_INITIAL (0.5f*MAX_VAR)

// ============== stereo & gradient calculation ======================
#define MIN_DEPTH 0.05f // this is the minimal depth tested for stereo.

// particularely important for initial pixel.
#define MAX_EPL_LENGTH_CROP 30.0f // maximum length of epl to search.
#define MIN_EPL_LENGTH_CROP (3.0f) // minimum length of epl to search.

// this is the distance of the sample points used for the stereo descriptor.
#define GRADIENT_SAMPLE_DIST 1.0f

// pixel a point needs to be away from border... if too small: segfaults!
#define SAMPLE_POINT_TO_BORDER 7

// pixels with too big an error are definitely thrown out.
#define MAX_ERROR_STEREO (1300.0f) // maximal photometric error for stereo to be successful (sum over 5 squared intensity differences)
#define MIN_DISTANCE_ERROR_STEREO (1.5f) // minimal multiplicative difference to second-best match to not be considered ambiguous.

// ============== Smoothing and regularization ======================
// distance factor for regularization.
// is used as assumed inverse depth variance between neighbouring pixel.
// basically determines the amount of spacial smoothing (small -> more smoothing).
#define REG_DIST_VAR (0.075f*0.075f*depthSmoothingFactor*depthSmoothingFactor)

// define how strict the merge-processes etc. are.
// are multiplied onto the difference, so the larger, the more restrictive.
#define DIFF_FAC_SMOOTHING (1.0f*1.0f)
#define DIFF_FAC_OBSERVE (1.0f*1.0f)
#define DIFF_FAC_PROP_MERGE (1.0f*1.0f)
#define DIFF_FAC_INCONSISTENT (1.0f * 1.0f)

// ============== initial stereo pixel selection ======================
#define MIN_EPL_GRAD_SQUARED (2.0f*2.0f)
#define MIN_EPL_LENGTH_SQUARED (1.0f*1.0f)
#define MIN_EPL_ANGLE_SQUARED (0.3f*0.3f)

// abs. grad at that location needs to be larger than this.
#define MIN_ABS_GRAD_CREATE (minUseGrad)
#define MIN_ABS_GRAD_DECREASE (minUseGrad)

// defines how large the stereo-search region is. it is [mean] +/- [std.dev]*STEREO_EPL_VAR_FAC
#define STEREO_EPL_VAR_FAC 2.0f
#define STEREO_EPL_VAR_FAC_OMNI 4.0f

}
