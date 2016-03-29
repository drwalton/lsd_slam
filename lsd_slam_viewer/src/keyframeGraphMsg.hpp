#ifndef KEYFRAMEGRAPHMSG_HPP_INCLUDED
#define KEYFRAMEGRAPHMSG_HPP_INCLUDED

#include <vector>

struct keyframeGraphMsg {
    unsigned int numFrames;
    std::vector<char> frameData;

    unsigned int numConstraints;
    std::vector<char> constraintsData;
};

#endif

