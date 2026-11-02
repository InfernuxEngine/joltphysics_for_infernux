// SPDX-License-Identifier: MIT
#pragma once

#include <Jolt/Physics/Constraints/ContactConstraintManager.h>
#include <Jolt/Physics/Constraints/MechanicalAssembly.h>
#include <Jolt/Physics/Constraints/CalculateSolverSteps.h>
#include <Jolt/Physics/Constraints/ConstraintPart/SpringPart.h>

JPH_NAMESPACE_BEGIN

class HingeConstraint;
class SliderConstraint;
class Constraint;

/// One island job owns all velocities and impulses of an armature mechanism.
/// Joint geometry is condensed once; every remaining row uses that same mass
/// response. No local two-body solver may write velocities during this solve.
class JointVelocitySolver : private MechanicalAssembly
{
public:
	JointVelocitySolver() : MechanicalAssembly(false) { }
	static bool NeedsResponse(Constraint **inConstraints, const uint32 *inBegin, const uint32 *inEnd);
	bool Solve(Constraint **inConstraints, const uint32 *inBegin, const uint32 *inEnd,
		ContactConstraintManager &ioContacts, const uint32 *inContactBegin, const uint32 *inContactEnd,
		float inDeltaTime, float inWarmStartRatio, CalculateSolverSteps &ioSteps);

private:
	using Jacobian = MechanicalResponse::Jacobian;
	struct Row
	{
		Jacobian mJacobian;
		MechanicalResponse::Impulse mResponse;
		double mLambda = 0, mPassive = 0, mBias = 0, mCompliance = 0;
		double mMin = -FLT_MAX, mMax = FLT_MAX, mDamping = 0, mFriction = 0;
		float *mLiveLambda = nullptr, *mLivePassive = nullptr;
		bool mDrive = false;
	};
	struct Contact
	{
		std::vector<Row> mNormals;
		std::vector<double> mDistances;
		Row mTangents[2], mAngular;
		MechanicalResponse::Plane mPlane;
		double mFriction;
	};
	struct Rotor { HingeConstraint *mHinge; Jacobian mJacobian; };

	void AddHinge(HingeConstraint &ioHinge, float inDeltaTime);
	void AddSlider(SliderConstraint &ioSlider, float inDeltaTime);
	bool SetSpring(Row &ioRow, float inDeltaTime, float inC, float inVelocityBias, const SpringSettings &inSpring);
	template <EMotionType Type1, EMotionType Type2>
	bool AddContact(ContactConstraintManager::ContactConstraintBase &ioContact);
	template <EMotionType Type1, EMotionType Type2>
	Row ContactRow(Body &ioBody1, Body &ioBody2, Vec3Arg inAxis, ContactConstraintPart<Type1, Type2> &ioPart, bool inPrepare = true);
	Row MakeRow(const Jacobian &inJacobian, float &ioLambda, bool inActive);
	double Candidate(const Row &inRow) const;
	bool Apply(Row &ioRow, double inLambda);
	bool SolveRow(Row &ioRow);
	bool Validate() const;

	std::vector<Row> mJointRows;
	std::vector<Contact> mContacts;
	std::vector<Rotor> mRotors;
};

JPH_NAMESPACE_END
