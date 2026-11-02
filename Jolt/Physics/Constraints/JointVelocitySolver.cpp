// SPDX-License-Identifier: MIT
#include <Jolt/Jolt.h>
#include <Jolt/Physics/Constraints/JointVelocitySolver.h>
#include <Jolt/Physics/Constraints/HingeConstraint.h>
#include <Jolt/Physics/Constraints/SliderConstraint.h>
#include <Jolt/Physics/Constraints/ConstraintPart/DriveImpulse.h>

JPH_NAMESPACE_BEGIN

bool JointVelocitySolver::NeedsResponse(Constraint **inConstraints, const uint32 *inBegin, const uint32 *inEnd)
{
	for (auto i = inBegin; i != inEnd; ++i)
		if (inConstraints[*i]->GetSubType() == EConstraintSubType::Hinge
			&& static_cast<HingeConstraint *>(inConstraints[*i])->GetArmature() > 0.0f)
			return true;
	return false;
}

size_t JointVelocitySolver::AddBody(Body &ioBody)
{
	const auto found = std::find(mBodies.begin(), mBodies.end(), &ioBody);
	if (found != mBodies.end()) return size_t(found - mBodies.begin());
	MechanicalResponse::BodyState state;
	state.mDynamic = ioBody.IsDynamic();
	const Vec3 v = ioBody.GetLinearVelocity(), w = ioBody.GetAngularVelocity();
	for (uint a = 0; a < 3; ++a) { state.mVelocity[a] = v[a]; state.mVelocity[a + 3] = w[a]; }
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

JointVelocitySolver::Jacobian JointVelocitySolver::Angular(Body &ioBody1, Body &ioBody2, Vec3Arg inAxis)
{
	Jacobian j{{AddBody(ioBody1), {}}, {AddBody(ioBody2), {}}};
	for (uint a = 0; a < 3; ++a) { j.mFirst.mJacobian[a + 3] = -inAxis[a]; j.mSecond.mJacobian[a + 3] = inAxis[a]; }
	return j;
}

JointVelocitySolver::Jacobian JointVelocitySolver::Linear(Body &ioBody1, Body &ioBody2, Vec3Arg inAxis, Vec3Arg inLever1, Vec3Arg inLever2)
{
	auto j = Angular(ioBody1, ioBody2, Vec3::sZero());
	for (uint a = 0; a < 3; ++a) {
		j.mFirst.mJacobian[a] = -inAxis[a]; j.mSecond.mJacobian[a] = inAxis[a];
		j.mFirst.mJacobian[a + 3] = -inLever1[a]; j.mSecond.mJacobian[a + 3] = inLever2[a];
	}
	return j;
}

JointVelocitySolver::Row JointVelocitySolver::MakeRow(const Jacobian &inJacobian, float &ioLambda, bool inActive)
{
	Row row;
	row.mJacobian = inJacobian;
	row.mLiveLambda = &ioLambda;
	row.mLambda = inActive? ioLambda : 0;
	return row;
}

bool JointVelocitySolver::SetSpring(Row &ioRow, float inDeltaTime, float inC, float inVelocityBias, const SpringSettings &inSpring)
{
	SpringPart spring;
	float effective_mass;
	const double k = ioRow.mResponse.mInverseEffectiveMass;
	if (inSpring.HasStiffnessOrDamping()) {
		// SI stiffness/damping remains defined at zero mobility. A frequency
		// or mass-normalized spring needs a finite effective physical mass.
		if (k == 0 && inSpring.mMode != ESpringMode::StiffnessAndDamping) return false;
		spring.CalculateSpringPropertiesWithSettings(inDeltaTime, float(k), inVelocityBias, inC, inSpring, effective_mass);
	} else
		spring.CalculateSpringPropertiesWithBias(inVelocityBias);
	ioRow.mCompliance = spring.GetSoftness();
	ioRow.mBias = spring.GetBias(0);
	return true;
}

void JointVelocitySolver::AddHinge(HingeConstraint &h, float dt)
{
	Body &a = *h.GetBody1(), &b = *h.GetBody2();
	for (const Vec3 axis : {Vec3::sAxisX(), Vec3::sAxisY(), Vec3::sAxisZ()})
		mResponse.AddEquation({Linear(a, b, axis, h.mPointConstraintPart.mR1.Cross(axis), h.mPointConstraintPart.mR2.Cross(axis))});
	for (const Vec3 axis : {h.mRotationConstraintPart.mB2xA1, h.mRotationConstraintPart.mC2xA1})
		mResponse.AddEquation({Angular(a, b, axis)});
	if (h.mArmature > 0) {
		auto j = Angular(a, b, h.mA1);
		mResponse.AddEquation({j, -h.mRotorSpeed, 1.0 / h.mArmature});
		mRotors.push_back({&h, j});
	}
}

void JointVelocitySolver::AddSlider(SliderConstraint &s, float dt)
{
	Body &a = *s.GetBody1(), &b = *s.GetBody2();
	for (const Vec3 axis : {s.mN1, s.mN2})
		mResponse.AddEquation({Linear(a, b, axis, (s.mR1 + s.mU).Cross(axis), s.mR2.Cross(axis))});
	for (const Vec3 axis : {Vec3::sAxisX(), Vec3::sAxisY(), Vec3::sAxisZ()})
		mResponse.AddEquation({Angular(a, b, axis)});
}

template <EMotionType T1, EMotionType T2>
JointVelocitySolver::Row JointVelocitySolver::ContactRow(Body &a, Body &b, Vec3Arg axis, ContactConstraintPart<T1, T2> &part)
{
	if (!part.IsActive()) return MakeRow(Angular(a, b, Vec3::sZero()), part.mTotalLambda, false);
	Vec3 r1 = Vec3::sZero(), r2 = Vec3::sZero();
	if constexpr (T1 != EMotionType::Static) r1 = Vec3::sLoadFloat3Unsafe(part.mR1PlusUxAxis);
	if constexpr (T2 != EMotionType::Static) r2 = Vec3::sLoadFloat3Unsafe(part.mR2xAxis);
	auto row = MakeRow(Linear(a, b, axis, r1, r2), part.mTotalLambda, part.IsActive());
	row.mBias = part.IsActive()? part.mBias : 0.0;
	row.mResponse = mResponse.Prepare(row.mJacobian);
	return row;
}

template <EMotionType T1, EMotionType T2>
bool JointVelocitySolver::AddContact(ContactConstraintManager::ContactConstraintBase &base)
{
	auto &c = static_cast<ContactConstraintManager::ContactConstraint<T1, T2> &>(base);
	Body &a = *c.mBody1, &b = *c.mBody2;
	// Per-contact artificial mass scales do not describe a single mechanical
	// mass operator. Never silently mix those with physical armature response.
	if ((a.IsDynamic() && (c.mInvMass1 != a.GetMotionProperties()->GetInverseMass() || c.mInvInertiaScale1 != 1))
		|| (b.IsDynamic() && (c.mInvMass2 != b.GetMotionProperties()->GetInverseMass() || c.mInvInertiaScale2 != 1))) return false;
	Contact contact;
	contact.mFriction = c.mCombinedFriction;
	Vec3 t1, t2; c.GetTangents(t1, t2);
	contact.mTangents[0] = ContactRow(a, b, t1, c.mFrictionConstraint1);
	contact.mTangents[1] = ContactRow(a, b, t2, c.mFrictionConstraint2);
	contact.mAngular = MakeRow(Angular(a, b, c.GetWorldSpaceNormal()), c.mAngularFrictionConstraint.mTotalLambda, c.mAngularFrictionConstraint.IsActive());
	contact.mAngular.mBias = c.mAngularFrictionConstraint.IsActive()? c.mAngularFrictionConstraint.mBias : 0.0;
	contact.mAngular.mResponse = mResponse.Prepare(contact.mAngular.mJacobian);
	// A deactivated row has no physical law, even if that axis is movable.
	if (!c.mFrictionConstraint1.IsActive()) contact.mTangents[0].mResponse = {};
	if (!c.mFrictionConstraint2.IsActive()) contact.mTangents[1].mResponse = {};
	if (!c.mAngularFrictionConstraint.IsActive()) contact.mAngular.mResponse = {};
	for (uint i = 0; i < c.mNumContactPoints; ++i) {
		auto row = ContactRow(a, b, c.GetWorldSpaceNormal(), c.mContactPoints[i].mNonPenetrationConstraint);
		row.mMin = 0;
		contact.mNormals.push_back(std::move(row));
		contact.mDistances.push_back(c.mCombinedFriction > 0? c.mContactPoints[i].mDistanceToFrictionCenter : 0.0);
	}
	mContacts.push_back(std::move(contact));
	return true;
}

double JointVelocitySolver::Candidate(const Row &row) const
{
	const double k = row.mResponse.mInverseEffectiveMass;
	return k > 0? row.mLambda - (mResponse.Velocity(row.mJacobian) + row.mBias + row.mCompliance * row.mLambda) / (k + row.mCompliance) : 0.0;
}

bool JointVelocitySolver::Apply(Row &row, double lambda)
{
	const double change = lambda - row.mLambda;
	row.mLambda = lambda;
	mResponse.Apply(row.mResponse, change);
	return change != 0;
}

bool JointVelocitySolver::SolveRow(Row &row)
{
	const double k = row.mResponse.mInverseEffectiveMass;
	if (k == 0) {
		// A locked mechanism still supports an applied motor torque through
		// bilateral reactions. It has zero velocity response, not zero force.
		const double q = mResponse.Velocity(row.mJacobian);
		if (row.mCompliance > 0) row.mLambda = std::clamp(-(q + row.mBias) / row.mCompliance, row.mMin, row.mMax);
		else if (row.mDrive) row.mLambda = q + row.mBias < 0? row.mMax : q + row.mBias > 0? row.mMin : std::clamp(0.0, row.mMin, row.mMax);
		else row.mLambda = std::clamp(0.0, row.mMin, row.mMax);
		row.mPassive = row.mDrive? -row.mDamping * q - (q > 0? row.mFriction : q < 0? -row.mFriction : 0) : 0;
		return false;
	}
	if (row.mDrive) {
		const double old_total = row.mLambda + row.mPassive;
		const auto result = SolveDriveImpulse(mResponse.Velocity(row.mJacobian) - k * old_total,
			k, row.mCompliance, row.mBias, row.mMin, row.mMax, row.mDamping, row.mFriction);
		const double delta = (result.mMotor - row.mLambda) + (result.mPassive - row.mPassive);
		row.mLambda = result.mMotor; row.mPassive = result.mPassive;
		mResponse.Apply(row.mResponse, delta);
		return delta != 0;
	}
	return Apply(row, std::clamp(Candidate(row), row.mMin, row.mMax));
}

bool JointVelocitySolver::Validate() const
{
	double scale = 1;
	for (size_t i = 0; i < mBodies.size(); ++i)
		for (double v : mResponse.GetBody(i).mVelocity) {
			if (!std::isfinite(v) || std::abs(v) > FLT_MAX) return false;
			scale = std::max(scale, std::abs(v));
		}
	return mResponse.GetMaxHardResidual() <= 1.0e-7 * scale;
}

bool JointVelocitySolver::Solve(Constraint **constraints, const uint32 *begin, const uint32 *end,
	ContactConstraintManager &manager, const uint32 *cbegin, const uint32 *cend,
	float dt, float ratio, CalculateSolverSteps &steps)
{
	// Capture bodies before Build, including free bodies joined only by contacts.
	for (auto i = cbegin; i != cend; ++i) {
		auto &c = *reinterpret_cast<ContactConstraintManager::ContactConstraintBase *>(manager.mConstraints + *i);
		AddBody(*c.mBody1); AddBody(*c.mBody2);
	}
	for (auto i = begin; i != end; ++i) {
		auto &c = *constraints[*i];
		steps(&c);
		switch (c.GetSubType()) {
		case EConstraintSubType::Hinge: AddHinge(static_cast<HingeConstraint &>(c), dt); break;
		case EConstraintSubType::Slider: AddSlider(static_cast<SliderConstraint &>(c), dt); break;
		default: return false;
		}
	}
	for (auto *body : mBodies) if (body->IsDynamic()) steps(body->GetMotionProperties());
	steps.Finalize();
	if (!mResponse.Build() || !Validate()) return false;
	for (auto i = begin; i != end; ++i) {
		if (constraints[*i]->GetSubType() == EConstraintSubType::Hinge) {
			auto &h = static_cast<HingeConstraint &>(*constraints[*i]);
			if (!h.mMotorConstraintPart.IsActive() && !h.mRotationLimitsConstraintPart.IsActive()) continue;
			const auto j = Angular(*h.GetBody1(), *h.GetBody2(), h.mA1);
			if (h.mMotorConstraintPart.IsActive()) {
				auto row = MakeRow(j, h.mMotorConstraintPart.mTotalLambda, true);
				row.mResponse = mResponse.Prepare(j);
				row.mDrive = true; row.mLivePassive = &h.mPassiveLambda; row.mPassive = h.mPassiveLambda;
				row.mDamping = dt * h.mViscousDamping; row.mFriction = dt * h.mMaxFrictionTorque;
				const bool driven = h.mMotorState == EMotorState::Velocity || (h.mMotorState != EMotorState::Off && h.mMotorSettings.mSpringSettings.HasStiffnessOrDamping());
				row.mMin = driven? dt * h.mMotorSettings.mMinTorqueLimit : 0;
				row.mMax = driven? dt * h.mMotorSettings.mMaxTorqueLimit : 0;
				if (h.mMotorState == EMotorState::Position || h.mMotorState == EMotorState::PositionAndVelocity)
					{ if (!SetSpring(row, dt, CenterAngleAroundZero(h.mTheta - h.mTargetAngle), h.mMotorState == EMotorState::PositionAndVelocity? -h.mTargetAngularVelocity : 0, h.mMotorSettings.mSpringSettings)) return false; }
				else row.mBias = h.mMotorState == EMotorState::Velocity? -h.mTargetAngularVelocity : 0;
				mJointRows.push_back(std::move(row));
			}
			if (h.mRotationLimitsConstraintPart.IsActive()) {
				auto row = MakeRow(j, h.mRotationLimitsConstraintPart.mTotalLambda, true);
				row.mResponse = mResponse.Prepare(j);
				if (h.mLimitsSpringSettings.HasStiffness() && !SetSpring(row, dt, h.GetSmallestAngleToLimit(), 0, h.mLimitsSpringSettings)) return false;
				if (h.mLimitsMin != h.mLimitsMax) { if (h.IsMinLimitClosest()) row.mMin = 0; else row.mMax = 0; }
				mJointRows.push_back(std::move(row));
			}
		} else {
			auto &s = static_cast<SliderConstraint &>(*constraints[*i]);
			if (!s.mMotorConstraintPart.IsActive() && !s.mPositionLimitsConstraintPart.IsActive()) continue;
			const auto j = Linear(*s.GetBody1(), *s.GetBody2(), s.mWorldSpaceSliderAxis, (s.mR1 + s.mU).Cross(s.mWorldSpaceSliderAxis), s.mR2.Cross(s.mWorldSpaceSliderAxis));
			if (s.mMotorConstraintPart.IsActive()) {
				auto row = MakeRow(j, s.mMotorConstraintPart.mTotalLambda, true); row.mResponse = mResponse.Prepare(j);
				if (s.mMotorState == EMotorState::Off) { row.mMin = -dt * s.mMaxFrictionForce; row.mMax = -row.mMin; }
				else { row.mMin = dt * s.mMotorSettings.mMinForceLimit; row.mMax = dt * s.mMotorSettings.mMaxForceLimit; }
				if (s.mMotorState == EMotorState::Position || s.mMotorState == EMotorState::PositionAndVelocity)
					{ if (!SetSpring(row, dt, s.mD - s.mTargetPosition, s.mMotorState == EMotorState::PositionAndVelocity? -s.mTargetVelocity : 0, s.mMotorSettings.mSpringSettings)) return false; }
				else row.mBias = s.mMotorState == EMotorState::Velocity? -s.mTargetVelocity : 0;
				mJointRows.push_back(std::move(row));
			}
			if (s.mPositionLimitsConstraintPart.IsActive()) {
				auto row = MakeRow(j, s.mPositionLimitsConstraintPart.mTotalLambda, true); row.mResponse = mResponse.Prepare(j);
				const bool below = s.mD <= s.mLimitsMin;
				if (s.mLimitsSpringSettings.HasStiffness() && !SetSpring(row, dt, s.mD - (below? s.mLimitsMin : s.mLimitsMax), 0, s.mLimitsSpringSettings)) return false;
				if (s.mLimitsMin != s.mLimitsMax) { if (below) row.mMin = 0; else row.mMax = 0; }
				mJointRows.push_back(std::move(row));
			}
		}
	}
	using Dispatch = bool (JointVelocitySolver::*)(ContactConstraintManager::ContactConstraintBase &);
	static const Dispatch table[3][3] = {
		{nullptr, nullptr, &JointVelocitySolver::AddContact<EMotionType::Static, EMotionType::Dynamic>},
		{nullptr, nullptr, &JointVelocitySolver::AddContact<EMotionType::Kinematic, EMotionType::Dynamic>},
		{&JointVelocitySolver::AddContact<EMotionType::Dynamic, EMotionType::Static>, &JointVelocitySolver::AddContact<EMotionType::Dynamic, EMotionType::Kinematic>, &JointVelocitySolver::AddContact<EMotionType::Dynamic, EMotionType::Dynamic>}
	};
	for (auto i = cbegin; i != cend; ++i) {
		auto &c = *reinterpret_cast<ContactConstraintManager::ContactConstraintBase *>(manager.mConstraints + *i);
		if (!(this->*table[int(c.mBody1->GetMotionType())][int(c.mBody2->GetMotionType())])(c)) return false;
	}
	auto warm = [&](Row &row) {
		row.mLambda *= ratio; row.mPassive *= ratio;
		mResponse.Apply(row.mResponse, row.mLambda + row.mPassive);
	};
	for (auto &row : mJointRows) warm(row);
	for (auto &c : mContacts) { for (auto &row : c.mNormals) warm(row); for (auto &row : c.mTangents) warm(row); warm(c.mAngular); }
	for (uint iteration = 0; iteration < steps.GetNumVelocitySteps(); ++iteration) {
		bool changed = false;
		for (auto &row : mJointRows) changed |= SolveRow(row);
		for (auto &c : mContacts) {
			double normal = 0, angular = 0;
			for (size_t i = 0; i < c.mNormals.size(); ++i) { normal += c.mNormals[i].mLambda; angular += c.mDistances[i] * c.mNormals[i].mLambda; }
			const double limit = c.mFriction * normal;
			double x = Candidate(c.mTangents[0]), y = Candidate(c.mTangents[1]);
			const double length = std::hypot(x, y);
			if (length > limit) { x *= limit / length; y *= limit / length; }
			changed |= Apply(c.mTangents[0], x); changed |= Apply(c.mTangents[1], y);
			c.mAngular.mMin = -c.mFriction * angular; c.mAngular.mMax = -c.mAngular.mMin;
			changed |= SolveRow(c.mAngular);
			for (auto &row : c.mNormals) changed |= SolveRow(row);
		}
		if (!changed) break;
	}
	if (!Validate()) return false;
	auto finite_row = [](const Row &r) { return std::isfinite(r.mLambda) && std::abs(r.mLambda) <= FLT_MAX && std::isfinite(r.mPassive) && std::abs(r.mPassive) <= FLT_MAX; };
	for (auto &r : mJointRows) if (!finite_row(r)) return false;
	for (auto &c : mContacts) { for (auto &r : c.mNormals) if (!finite_row(r)) return false; for (auto &r : c.mTangents) if (!finite_row(r)) return false; if (!finite_row(c.mAngular)) return false; }
	// Publish once. Original contact-cache and diagnostic stream consume these
	// live lambdas, not a parallel private impulse cache.
	for (size_t i = 0; i < mBodies.size(); ++i) if (mBodies[i]->IsDynamic()) {
		const auto &v = mResponse.GetBody(i).mVelocity;
		auto *mp = mBodies[i]->GetMotionProperties();
		mp->ApplyLinearVelocityStep(Vec3(float(v[0]), float(v[1]), float(v[2])));
		mp->ApplyAngularVelocityStep(Vec3(float(v[3]), float(v[4]), float(v[5])));
	}
	auto publish = [](Row &r) { if (r.mLiveLambda) *r.mLiveLambda = float(r.mLambda); if (r.mLivePassive) *r.mLivePassive = float(r.mPassive); };
	for (auto &r : mJointRows) publish(r);
	for (auto &c : mContacts) { for (auto &r : c.mNormals) publish(r); for (auto &r : c.mTangents) publish(r); publish(c.mAngular); }
	for (auto &r : mRotors) r.mHinge->mRotorSpeed = mResponse.Velocity(r.mJacobian);
	return true;
}

JPH_NAMESPACE_END
