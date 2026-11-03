// SPDX-License-Identifier: MIT
#pragma once

#include <algorithm>
#include <cmath>

JPH_NAMESPACE_BEGIN

template <class T> struct DriveImpulse
{
	T mMotor, mPassive, mTotal, mRotor, mVelocity;
};

/// Exact scalar solve for a capped motor, uncapped viscous damping and dry
/// friction. Arguments and results are impulses for the current substep.
template <class T>
inline DriveImpulse<T> SolveDriveImpulse(T inFreeVelocity, T inInverseMass, T inCompliance, T inBias,
	T inMinMotor, T inMaxMotor, T inDamping, T inFriction, T inArmature, T inOldRotorSpeed)
{
	const T shifted = inFreeVelocity + inInverseMass * inArmature * inOldRotorSpeed;
	T motor;
	if (inCompliance > T(0)) motor = std::clamp(-inBias / inCompliance, inMinMotor, inMaxMotor);
	else if (inBias < T(0)) motor = inMaxMotor;
	else if (inBias > T(0)) motor = inMinMotor;
	else motor = std::clamp(-shifted / inInverseMass, inMinMotor, inMaxMotor);
	const T at_rest = shifted + inInverseMass * motor;
	if (std::abs(at_rest) <= inInverseMass * inFriction) {
		const T total = -inFreeVelocity/inInverseMass;
		const T rotor = inArmature*inOldRotorSpeed;
		return {motor,total-motor-rotor,total,rotor,T(0)};
	}
	const T friction = at_rest > T(0)? inFriction : -inFriction;
	const T b = T(1) + inInverseMass * (inDamping+inArmature);
	motor = std::clamp(-(shifted - inInverseMass * friction + b * inBias)
		/ (b * inCompliance + inInverseMass), inMinMotor, inMaxMotor);
	// Form the net impulse without subtracting two almost equal velocities
	// and multiplying that rounding error by a potentially large armature.
	const T total = (motor-friction-inDamping*inFreeVelocity
		+inArmature*(inOldRotorSpeed-inFreeVelocity))/b;
	const T velocity = inFreeVelocity+inInverseMass*total;
	const T passive = -inDamping*velocity-friction;
	const T rotor = (inArmature/b)*(inOldRotorSpeed-inFreeVelocity
		+inInverseMass*(inDamping*inOldRotorSpeed-motor+friction));
	return {motor,passive,total,rotor,velocity};
}

JPH_NAMESPACE_END
