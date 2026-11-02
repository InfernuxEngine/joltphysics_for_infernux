// SPDX-License-Identifier: MIT
#pragma once

#include <Jolt/Math/Quat.h>
#include <array>
#include <cmath>

JPH_NAMESPACE_BEGIN
namespace JointGeometry
{
struct AngularRow { Vec3 mAxis; double mError; };

/// Derivatives of two spherical coordinates in body 1's attached frame.
/// The caller chooses/fixes n,c for this linearization, with |y| >= |z|;
/// then d*d+y*y >= 1/2, including opposite hinge axes.
inline std::array<AngularRow, 2> HingeAlignment(Vec3Arg a, Vec3Arg b, Vec3Arg n, Vec3Arg c)
{
	const double d = a.Dot(b), y = n.Dot(b), z = c.Dot(b);
	const double r2 = d*d + y*y, r = std::sqrt(r2);
	const Vec3 jd = b.Cross(a), jy = b.Cross(n), jz = b.Cross(c);
	return {{
		{float(d/r2)*jy - float(y/r2)*jd, std::atan2(y,d)},
		{float(r/(r2+z*z))*jz - float(z/r/(r2+z*z))*(float(d)*jd + float(y)*jy), std::atan2(z,r)}
	}};
}

/// Body-1-local rotation logarithm and its exact left-perturbation derivative.
/// The shortest quaternion branch is fixed by the current pose.
inline std::array<AngularRow, 3> FixedRotation(QuatArg q1, QuatArg q2, QuatArg inverseInitial)
{
	Quat q = (q1.Conjugated()*q2*inverseInitial).Normalized();
	if (q.GetW() < 0) q = -q;
	const Vec3 v = q.GetXYZ();
	const double sine = v.Length(), theta = 2*std::atan2(sine,double(q.GetW()));
	const Vec3 phi = float(sine > 0? theta/sine : 2)*v;
	const double theta2 = theta*theta;
	const double coefficient = theta2 < 1.0e-6? 1.0/12 + theta2/720 + theta2*theta2/30240
		: (1 - .5*theta*q.GetW()/sine)/theta2;
	std::array<AngularRow,3> result;
	for (uint i = 0; i < 3; ++i) {
		const Vec3 unit = i == 0? Vec3::sAxisX() : i == 1? Vec3::sAxisY() : Vec3::sAxisZ();
		const Vec3 local = unit - .5f*unit.Cross(phi) + float(coefficient)*(phi[i]*phi - float(theta2)*unit);
		result[i] = {q1*local, phi[i]};
	}
	return result;
}

/// Twist about the reference hinge axis. A pi swing has no defined twist;
/// no arbitrary physical axis is substituted at that singular configuration.
inline bool HingeTwist(QuatArg q1, QuatArg q2, QuatArg inverseInitial, Vec3Arg localAxis, AngularRow &outRow)
{
	const Quat q = (q1.Conjugated()*q2*inverseInitial).Normalized();
	const Vec3 v = q.GetXYZ();
	const double w = q.GetW(), p = v.Dot(localAxis), denominator = w*w + p*p;
	if (!(denominator > 0)) return false;
	outRow.mAxis = q1*(float(w*w/denominator)*localAxis + float(w/denominator)*v.Cross(localAxis) + float(p/denominator)*v);
	outRow.mError = CenterAngleAroundZero(float(2*std::atan2(p,w)));
	return true;
}
}
JPH_NAMESPACE_END
