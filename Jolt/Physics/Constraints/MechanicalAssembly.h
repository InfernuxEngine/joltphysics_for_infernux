// SPDX-License-Identifier: MIT
#pragma once

#include <Jolt/Physics/Body/Body.h>
#include <Jolt/Physics/Constraints/MechanicalResponse.h>

JPH_NAMESPACE_BEGIN

/// Native body identities and masked mass shared by the two solver phases.
/// Displacement scratch never imports real velocities or rotor momentum.
class MechanicalAssembly
{
protected:
	using Jacobian = MechanicalResponse::Jacobian;
	explicit MechanicalAssembly(bool inDisplacement) : mDisplacement(inDisplacement) { }

	size_t AddBody(Body &ioBody)
	{
		const auto found = std::find(mBodies.begin(), mBodies.end(), &ioBody);
		if (found != mBodies.end()) return size_t(found - mBodies.begin());
		MechanicalResponse::BodyState state;
		state.mDynamic = ioBody.IsDynamic();
		if (!mDisplacement) {
			const Vec3 v = ioBody.GetLinearVelocity(), w = ioBody.GetAngularVelocity();
			for (uint a = 0; a < 3; ++a) { state.mVelocity[a] = v[a]; state.mVelocity[a + 3] = w[a]; }
		}
		if (state.mDynamic) {
			const auto *mp = ioBody.GetMotionProperties();
			const Vec3 mask = mp->LockTranslation(Vec3::sReplicate(1.0f));
			const Mat44 inertia = mp->GetInverseInertiaForRotation(Mat44::sRotation(ioBody.GetRotation()));
			for (uint a = 0; a < 3; ++a) {
				state.mInverseMass[a * 6 + a] = mp->GetInverseMass() * mask[a];
				for (uint b = 0; b < 3; ++b) state.mInverseMass[(a + 3) * 6 + b + 3] = inertia(a, b);
			}
		}
		mBodies.push_back(&ioBody);
		return mResponse.AddBody(state);
	}

	Jacobian Angular(Body &ioBody1, Body &ioBody2, Vec3Arg inAxis)
	{
		Jacobian j{{AddBody(ioBody1), {}}, {AddBody(ioBody2), {}}};
		for (uint a = 0; a < 3; ++a) { j.mFirst.mJacobian[a + 3] = -inAxis[a]; j.mSecond.mJacobian[a + 3] = inAxis[a]; }
		return j;
	}

	Jacobian Linear(Body &ioBody1, Body &ioBody2, Vec3Arg inAxis, Vec3Arg inLever1, Vec3Arg inLever2)
	{
		auto j = Angular(ioBody1, ioBody2, Vec3::sZero());
		for (uint a = 0; a < 3; ++a) {
			j.mFirst.mJacobian[a] = -inAxis[a]; j.mSecond.mJacobian[a] = inAxis[a];
			j.mFirst.mJacobian[a + 3] = -inLever1[a]; j.mSecond.mJacobian[a + 3] = inLever2[a];
		}
		return j;
	}

	MechanicalResponse mResponse;
	std::vector<Body *> mBodies;

private:
	bool mDisplacement;
};

JPH_NAMESPACE_END
