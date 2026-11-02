// SPDX-License-Identifier: MIT
#pragma once

#include <algorithm>
#include <cmath>

JPH_NAMESPACE_BEGIN

template <class T> struct DriveImpulse
{
	T mMotor, mPassive;
};

/// Exact scalar solve for a capped motor, uncapped viscous damping and dry
/// friction. Arguments and results are impulses for the current substep.
template <class T>
inline DriveImpulse<T> SolveDriveImpulse(T inFreeVelocity, T inInverseMass, T inCompliance, T inBias,
	T inMinMotor, T inMaxMotor, T inDamping, T inFriction)
{
	T motor;
	if (inCompliance > T(0)) motor = std::clamp(-inBias / inCompliance, inMinMotor, inMaxMotor);
	else if (inBias < T(0)) motor = inMaxMotor;
	else if (inBias > T(0)) motor = inMinMotor;
	else motor = std::clamp(-inFreeVelocity / inInverseMass, inMinMotor, inMaxMotor);
	const T at_rest = inFreeVelocity + inInverseMass * motor;
	if (std::abs(at_rest) <= inInverseMass * inFriction)
		return {motor, -inFreeVelocity / inInverseMass - motor};
	const T friction = at_rest > T(0)? inFriction : -inFriction;
	const T b = T(1) + inInverseMass * inDamping;
	motor = std::clamp(-(inFreeVelocity - inInverseMass * friction + b * inBias)
		/ (b * inCompliance + inInverseMass), inMinMotor, inMaxMotor);
	const T velocity = (inFreeVelocity + inInverseMass * motor - inInverseMass * friction) / b;
	return {motor, -inDamping * velocity - friction};
}

JPH_NAMESPACE_END
