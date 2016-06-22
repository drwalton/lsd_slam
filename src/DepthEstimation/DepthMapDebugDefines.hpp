#pragma once

// *** Stereo related #defines ***
#define DEBUG_SAVE_VAR_IMS 0
#define DEBUG_SAVE_IDEPTH_IMS 0
#define DEBUG_SAVE_MATCH_IMS 0
#define DEBUG_SAVE_SEARCH_RANGE_IMS 1
#define DEBUG_SAVE_RESULT_IMS 1
#define DEBUG_SAVE_PIXEL_DISPARITY_IMS 0
#define DEBUG_SAVE_EPL_LENGTH_IMS 0

// *** Var calculation related #defines ***
//Save visualisations of components of variance
#define DEBUG_SAVE_GRAD_ALONG_LINE_IMS 0
#define DEBUG_SAVE_GEO_DISP_ERROR_IMS 0
#define DEBUG_SAVE_PHOTO_DISP_ERROR_IMS 0
#define DEBUG_SAVE_DISCRETIZATION_ERROR_IMS 0

// *** Point cloud related #defines ***
//Save a cloud showing the raw output of stereo on each
// new frame.
#define DEBUG_SAVE_FRAME_STEREO_POINT_CLOUDS 0
//Save a cloud showing the new state of the keyframe
// depth map after performing stereo with each new frame,
// and updating.
#define DEBUG_SAVE_KEYFRAME_POINT_CLOUDS_EACH_FRAME 0
//Save two clouds, showing each keyframe before and after
// the depth propagation process.
#define DEBUG_SAVE_KEYFRAME_PROPAGATION_CLOUDS 0
