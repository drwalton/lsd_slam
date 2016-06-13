#include "Util/ImgProc.hpp"

//This typedef hack is designed to fix an issue with both libtiff and Eigen3
// defining int64 and uint64 typedefs. This renames all the libtiff typedefs
// to int64_ignore, and uint64_ignore, instead.
#ifdef __clang__
#define int64 int64_ignore
#define uint64 uint64_ignore
#endif

#include <tiffio.h>

//Undefining again - from now on, int64/uint64 use the Eigen3 typedefs.
#ifdef __clang__
#undef int64
#undef uint64
#endif

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

bool imwriteFloat(const std::string &filename, const cv::Mat &imToSave)
{
	if (imToSave.type() != CV_32FC1) {
		throw std::runtime_error("Can only use this function to save 32-bit floating point images!");
	}

	TIFF *tif = TIFFOpen(filename.c_str(), "w");

	if (tif) {
		TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, imToSave.cols);
		TIFFSetField(tif, TIFFTAG_IMAGELENGTH, imToSave.rows);
		TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, 1);
		TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, 32);
		TIFFSetField(tif, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);

		for (size_t row = 0; row < size_t(imToSave.rows); ++row) {
			//Write each row of the image in turn.
			float *rptr = const_cast<float*>(imToSave.ptr<float>(row));
			if (TIFFWriteScanline(tif, rptr, row, 0) < 0) {
				//Error writing row
				std::cout << "TIFF writing error!!!" << std::endl;
				TIFFClose(tif);
				return false;
			}
		}
	}

	TIFFClose(tif);
	return true;
}
cv::Mat imreadFloat(const std::string &filename)
{
	TIFF *tif = TIFFOpen(filename.c_str(), "r");
	if (tif) {
		uint32 w, h;
		uint16 samplesPerPixel, bitsPerSample;
		TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &w);
		TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &h);

		TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &samplesPerPixel);
		TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bitsPerSample);
		if (samplesPerPixel != 1 || bitsPerSample != 32) {
			std::cout << "WRONG IMAGE FORMAT!";
			TIFFClose(tif);
			return cv::Mat();
		}
		cv::Mat im(cv::Size(w, h), CV_32FC1);
		for (size_t row = 0; row < h; ++row) {
			float *rptr = im.ptr<float>(row);
			if (TIFFReadScanline(tif, rptr, row) < 0) {
				std::cout << "UNABLE TO READ SCANLINE IN IMAGE";
				break;
			}
		}
		TIFFClose(tif);
		return im;
	}
	return cv::Mat();
}

}
