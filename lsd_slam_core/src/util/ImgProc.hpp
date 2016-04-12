#ifndef UTIL_IMGPROC_HPP_INCLUDED
#define UTIL_IMGPROC_HPP_INCLUDED

namespace lsd_slam
{

void downscaleImageHalf(const float *in, float *out, size_t w, size_t h) {
	for (size_t r = 0; r < h; r+=2) {
		for (size_t c = 0; c < w; c+=2) {
			*out = 0.25f *
				in[r*w + c] +
				in[(r + 1)*w + c] +
				in[r*w + c + 1] +
				in[(r + 1)*w + c + 1];
			++out;
		}
	}
}

}

#endif
