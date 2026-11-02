// SPDX-License-Identifier: MIT
#pragma once

#include <Jolt/Physics/Constraints/MechanicalAssembly.h>
#include <Jolt/Physics/Constraints/ContactConstraintManager.h>

JPH_NAMESPACE_BEGIN
class Constraint;
class HingeConstraint;
class SliderConstraint;
struct PhysicsSettings;

/// One nonlinear position iteration. Scratch displacements never become
/// body velocity, rotor momentum or warm-start impulses.
class JointPositionSolver : private MechanicalAssembly
{
public:
	enum class EResult { Applied, Unchanged, Invalid };
	static EResult Solve(Constraint **inConstraints, const uint32 *inBegin, const uint32 *inEnd,
		ContactConstraintManager &ioContacts, const uint32 *inContactBegin, const uint32 *inContactEnd,
		const PhysicsSettings &inSettings);

private:
	struct Pose { RVec3 mPosition; Quat mRotation; };
	struct Joint { Constraint *mConstraint; bool mTwistDefined; bool mSwapHingeChart; };
	struct Contact { size_t mBody1, mBody2; Vec3 mLocal1, mLocal2, mNormal; };
	struct Geometry
	{
		Jacobian mJacobian;
		double mError;
		bool mUnilateral = false;
		double mMaxCorrection = DBL_MAX;
	};
	struct Row { Jacobian mJacobian; MechanicalResponse::Impulse mResponse; double mBias, mLambda = 0; };

	explicit JointPositionSolver(const PhysicsSettings &inSettings) : MechanicalAssembly(true), mSettings(inSettings) { }
	EResult Run(Constraint **inConstraints, const uint32 *inBegin, const uint32 *inEnd,
		ContactConstraintManager &ioContacts, const uint32 *inContactBegin, const uint32 *inContactEnd);
	template <EMotionType Type1, EMotionType Type2>
	bool CollectContacts(ContactConstraintManager::ContactConstraintBase &inContact, ContactConstraintManager &inManager);
	bool GeometryAt(const std::vector<Pose> &inPoses, std::vector<Geometry> &outRows);
	void HingeGeometry(const Joint &inJoint, const std::vector<Pose> &inPoses, std::vector<Geometry> &outRows, bool &outValid);
	void SliderGeometry(SliderConstraint &inJoint, const std::vector<Pose> &inPoses, std::vector<Geometry> &outRows);
	static double Merit(const std::vector<Geometry> &inRows);
	void Hard(std::vector<Geometry> &out, const Jacobian &j, double error) { out.push_back({j, error}); }
	void Boundary(std::vector<Geometry> &out, const Jacobian &j, double error, double cap = DBL_MAX)
	{
		out.push_back({j, error, true, cap});
	}
	static Jacobian Negated(Jacobian j);
	Pose CurrentPose(Body &inBody) const { return {inBody.GetCenterOfMassPosition(), inBody.GetRotation()}; }

	const PhysicsSettings &mSettings;
	std::vector<Joint> mJoints;
	std::vector<Contact> mContacts;
	std::vector<Pose> mPoses;
};
JPH_NAMESPACE_END
