#ifndef MATH_KAHANVAL_HPP_INCLUDED
#define MATH_KAHANVAL_HPP_INCLUDED

#include "VectorTypes.hpp"

namespace lsd_slam
{

/// \brief This class template wraps Kahan summation of a value.
/// Kahan summation is a summation algorithm designed to reduce the numerical
/// errors inherent in sequentially adding large numbers of small floating
/// point values.
/// This class has an overloaded `+=` operator. Note that other arithmetic
/// operators are not overloaded.
template <typename T> class KahanVal
{
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
public:
	KahanVal(T initVal = T(0))
		:val(initVal),
		c(static_cast<T>(0)),
		y(static_cast<T>(0)),
		t(static_cast<T>(0)) {};
	const T& get() const { return val; };
	void set(const T& val) { this->val = val; };
	KahanVal& operator += (T rhs)
	{
		y = rhs - c;
		t = val + y;
		c = (t - val) - y;
		val = t;

		return *this;
	};
private:
	T val, c, y, t;
};

#ifndef __APPLE__
template<> KahanVal<mat3>::KahanVal(mat3 initVal);
template<> KahanVal<mat3>::KahanVal(mat3 initVal)
	:val(initVal), c(mat3::Zero()), y(mat3::Zero()), t(mat3::Zero()) {}
template<> KahanVal<vec3>::KahanVal(vec3 initVal);
template<> KahanVal<vec3>::KahanVal<vec3>(vec3 initVal)
: val(initVal), c(Eigen::Vector3f::Zero()),
y(Eigen::Vector3f::Zero()), t(Eigen::Vector3f::Zero()) {}
#else
template<> KahanVal<mat3>::KahanVal(mat3 initVal);
template<> KahanVal<vec3>::KahanVal(vec3 initVal);

#endif

#endif

}
