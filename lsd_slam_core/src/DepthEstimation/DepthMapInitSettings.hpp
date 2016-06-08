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


}
