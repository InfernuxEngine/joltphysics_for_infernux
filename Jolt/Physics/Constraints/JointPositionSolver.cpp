// SPDX-License-Identifier: MIT
#include <Jolt/Jolt.h>
#include <Jolt/Physics/Constraints/JointPositionSolver.h>
#include <Jolt/Physics/Constraints/JointGeometry.h>
#include <Jolt/Physics/Constraints/HingeConstraint.h>
#include <Jolt/Physics/Constraints/SliderConstraint.h>
#include <Jolt/Physics/Constraints/CalculateSolverSteps.h>

JPH_NAMESPACE_BEGIN

JointPositionSolver::Jacobian JointPositionSolver::Negated(Jacobian j)
{
	for (auto *term : {&j.mFirst, &j.mSecond})
		for (double &value : term->mJacobian) value = -value;
	return j;
}

void JointPositionSolver::HingeGeometry(const Joint &joint, const std::vector<Pose> &poses, std::vector<Geometry> &rows, bool &valid)
{
	auto &h = static_cast<HingeConstraint &>(*joint.mConstraint);
	Body &a = *h.GetBody1(), &b = *h.GetBody2();
	const auto &p1 = poses[AddBody(a)], &p2 = poses[AddBody(b)];
	const Vec3 r1 = p1.mRotation*h.mLocalSpacePosition1, r2 = p2.mRotation*h.mLocalSpacePosition2;
	const Vec3 error = Vec3(p2.mPosition - p1.mPosition) + r2 - r1;
	for (uint i = 0; i < 3; ++i) {
		const Vec3 axis = i == 0? Vec3::sAxisX() : i == 1? Vec3::sAxisY() : Vec3::sAxisZ();
		Hard(rows, Linear(a,b,axis,r1.Cross(axis),r2.Cross(axis)), error[i]);
	}
	const Vec3 axis1 = p1.mRotation*h.mLocalSpaceHingeAxis1, axis2 = p2.mRotation*h.mLocalSpaceHingeAxis2;
	Vec3 normal = p1.mRotation*h.mLocalSpaceNormalAxis1;
	normal = (normal - axis1.Dot(normal)*axis1).Normalized();
	Vec3 other = axis1.Cross(normal);
	if (joint.mSwapHingeChart) std::swap(normal,other);
	const auto alignment = JointGeometry::HingeAlignment(axis1,axis2,normal,other);
	for (const auto &row : alignment) Hard(rows,Angular(a,b,row.mAxis),row.mError);
	if (h.mHasLimits && !h.mLimitsSpringSettings.HasStiffness() && joint.mTwistDefined) {
		JointGeometry::AngularRow twist;
		if (!JointGeometry::HingeTwist(p1.mRotation,p2.mRotation,h.mInvInitialOrientation,h.mLocalSpaceHingeAxis1,twist)) { valid = false; return; }
		const auto j = Angular(a,b,twist.mAxis);
		const double lower = CenterAngleAroundZero(float(twist.mError-h.mLimitsMin));
		const double upper = CenterAngleAroundZero(float(twist.mError-h.mLimitsMax));
		if (h.mLimitsMin == h.mLimitsMax) Hard(rows,j,lower);
		else if (twist.mError < h.mLimitsMin || twist.mError > h.mLimitsMax) {
			// An excluded arc crosses the +/-pi seam. Correct through the
			// nearest endpoint rather than driving the long way around.
			if (std::abs(lower) < std::abs(upper)) Boundary(rows,j,lower);
			else Boundary(rows,Negated(j),-upper);
		} else {
			Boundary(rows,j,twist.mError-h.mLimitsMin);
			Boundary(rows,Negated(j),h.mLimitsMax-twist.mError);
		}
	}
}

void JointPositionSolver::SliderGeometry(SliderConstraint &s, const std::vector<Pose> &poses, std::vector<Geometry> &rows)
{
	Body &a = *s.GetBody1(), &b = *s.GetBody2();
	const auto &p1 = poses[AddBody(a)], &p2 = poses[AddBody(b)];
	const Vec3 r1 = p1.mRotation*s.mLocalSpacePosition1, r2 = p2.mRotation*s.mLocalSpacePosition2;
	const Vec3 u = Vec3(p2.mPosition-p1.mPosition)+r2-r1;
	for (const Vec3 local : {s.mLocalSpaceNormal1,s.mLocalSpaceNormal2}) {
		const Vec3 axis = p1.mRotation*local;
		Hard(rows,Linear(a,b,axis,(r1+u).Cross(axis),r2.Cross(axis)),u.Dot(axis));
	}
	for (const auto &row : JointGeometry::FixedRotation(p1.mRotation,p2.mRotation,s.mInvInitialOrientation))
		Hard(rows,Angular(a,b,row.mAxis),row.mError);
	if (s.mHasLimits && !s.mLimitsSpringSettings.HasStiffness()) {
		const Vec3 axis = p1.mRotation*s.mLocalSpaceSliderAxis1;
		const auto j = Linear(a,b,axis,(r1+u).Cross(axis),r2.Cross(axis));
		const double distance = u.Dot(axis);
		if (s.mLimitsMin == s.mLimitsMax) Hard(rows,j,distance-s.mLimitsMin);
		else {
			Boundary(rows,j,distance-s.mLimitsMin);
			Boundary(rows,Negated(j),s.mLimitsMax-distance);
		}
	}
}

bool JointPositionSolver::GeometryAt(const std::vector<Pose> &poses, std::vector<Geometry> &rows)
{
	bool valid = true;
	for (const auto &joint : mJoints)
		if (joint.mConstraint->GetSubType() == EConstraintSubType::Hinge) HingeGeometry(joint,poses,rows,valid);
		else SliderGeometry(static_cast<SliderConstraint &>(*joint.mConstraint),poses,rows);
	for (const auto &c : mContacts) {
		const auto &p1 = poses[c.mBody1], &p2 = poses[c.mBody2];
		const Vec3 r1 = p1.mRotation*c.mLocal1, r2 = p2.mRotation*c.mLocal2;
		const double separation = (Vec3(p2.mPosition-p1.mPosition)+r2-r1).Dot(c.mNormal)+mSettings.mPenetrationSlop;
		Boundary(rows,Linear(*mBodies[c.mBody1],*mBodies[c.mBody2],c.mNormal,r1.Cross(c.mNormal),r2.Cross(c.mNormal)),separation,mSettings.mMaxPenetrationDistance);
	}
	for (const auto &row : rows) {
		valid &= std::isfinite(row.mError);
		for (const auto *term : {&row.mJacobian.mFirst,&row.mJacobian.mSecond})
			for (double value : term->mJacobian) valid &= std::isfinite(value);
	}
	return valid;
}

double JointPositionSolver::Merit(const std::vector<Geometry> &rows)
{
	double result = 0;
	for (const auto &row : rows) result += Square(row.mUnilateral? std::min(row.mError,0.0) : row.mError);
	return result;
}

template <EMotionType T1, EMotionType T2>
bool JointPositionSolver::CollectContacts(ContactConstraintManager::ContactConstraintBase &base, ContactConstraintManager &manager)
{
	auto &c = static_cast<ContactConstraintManager::ContactConstraint<T1,T2> &>(base);
	Body &a = *c.mBody1, &b = *c.mBody2;
	if ((a.IsDynamic() && (c.mInvMass1 != a.GetMotionProperties()->GetInverseMass() || c.mInvInertiaScale1 != 1))
		|| (b.IsDynamic() && (c.mInvMass2 != b.GetMotionProperties()->GetInverseMass() || c.mInvInertiaScale2 != 1))) return false;
	const size_t first = AddBody(a), second = AddBody(b);
	const auto &cached = manager.mWriteCache->FromHandle(c.mCachedManifoldHandle)->GetValue();
	for (uint i = 0; i < c.mNumContactPoints; ++i)
		mContacts.push_back({first,second,Vec3::sLoadFloat3Unsafe(cached.mContactPoints[i].mPosition1),Vec3::sLoadFloat3Unsafe(cached.mContactPoints[i].mPosition2),c.GetWorldSpaceNormal()});
	return true;
}

JointPositionSolver::EResult JointPositionSolver::Run(Constraint **constraints, const uint32 *begin, const uint32 *end,
	ContactConstraintManager &manager, const uint32 *cbegin, const uint32 *cend)
{
	CalculateSolverSteps steps(mSettings);
	for (auto i = begin; i != end; ++i) {
		auto &c = *constraints[*i];
		if (c.GetSubType() != EConstraintSubType::Hinge && c.GetSubType() != EConstraintSubType::Slider) return EResult::Invalid;
		auto &two = static_cast<TwoBodyConstraint &>(c);
		AddBody(*two.GetBody1()); AddBody(*two.GetBody2()); steps(&c);
		bool twist_defined = true, swap_chart = false;
		if (c.GetSubType() == EConstraintSubType::Hinge) {
			auto &h = static_cast<HingeConstraint &>(c);
			JointGeometry::AngularRow unused;
			twist_defined = JointGeometry::HingeTwist(h.GetBody1()->GetRotation(),h.GetBody2()->GetRotation(),h.mInvInitialOrientation,h.mLocalSpaceHingeAxis1,unused);
			const Vec3 axis = h.GetBody1()->GetRotation()*h.mLocalSpaceHingeAxis1;
			Vec3 normal = h.GetBody1()->GetRotation()*h.mLocalSpaceNormalAxis1;
			normal = (normal-axis.Dot(normal)*axis).Normalized();
			const Vec3 other = axis.Cross(normal), b = h.GetBody2()->GetRotation()*h.mLocalSpaceHingeAxis2;
			swap_chart = std::abs(other.Dot(b)) > std::abs(normal.Dot(b));
		}
		// A singular off-manifold twist has no limit coordinate yet. Correct
		// the swing first; the next outer iteration rebuilds the limit rows.
		mJoints.push_back({&c,twist_defined,swap_chart});
	}
	using Dispatch = bool (JointPositionSolver::*)(ContactConstraintManager::ContactConstraintBase &,ContactConstraintManager &);
	static const Dispatch table[3][3] = {
		{nullptr,nullptr,&JointPositionSolver::CollectContacts<EMotionType::Static,EMotionType::Dynamic>},
		{nullptr,nullptr,&JointPositionSolver::CollectContacts<EMotionType::Kinematic,EMotionType::Dynamic>},
		{&JointPositionSolver::CollectContacts<EMotionType::Dynamic,EMotionType::Static>,&JointPositionSolver::CollectContacts<EMotionType::Dynamic,EMotionType::Kinematic>,&JointPositionSolver::CollectContacts<EMotionType::Dynamic,EMotionType::Dynamic>}
	};
	for (auto i = cbegin; i != cend; ++i) {
		auto &c = *reinterpret_cast<ContactConstraintManager::ContactConstraintBase *>(manager.mConstraints+*i);
		if (!(this->*table[int(c.mBody1->GetMotionType())][int(c.mBody2->GetMotionType())])(c,manager)) return EResult::Invalid;
	}
	for (auto *body : mBodies) {
		mPoses.push_back(CurrentPose(*body));
		if (body->IsDynamic()) steps(body->GetMotionProperties());
	}
	steps.Finalize();
	std::vector<Geometry> geometry;
	if (!GeometryAt(mPoses,geometry)) return EResult::Invalid;
	const double initial = Merit(geometry);
	if (!std::isfinite(initial)) return EResult::Invalid;
	if (initial == 0) return EResult::Unchanged;
	for (const auto &row : geometry)
		if (!row.mUnilateral) mResponse.AddEquation({row.mJacobian,mSettings.mBaumgarte*row.mError});
	for (const auto &joint : mJoints)
		if (joint.mConstraint->GetSubType() == EConstraintSubType::Hinge) {
			auto &h = static_cast<HingeConstraint &>(*joint.mConstraint);
			if (h.mArmature > 0)
				mResponse.AddEquation({Angular(*h.GetBody1(),*h.GetBody2(),h.GetBody1()->GetRotation()*h.mLocalSpaceHingeAxis1),0,1.0/h.mArmature});
		}
	if (!mResponse.BuildPosition()) return EResult::Invalid;
	std::vector<Row> rows;
	for (const auto &g : geometry) if (g.mUnilateral) {
		const double bias = g.mError < 0? mSettings.mBaumgarte*std::max(g.mError,-g.mMaxCorrection) : g.mError;
		rows.push_back({g.mJacobian,mResponse.Prepare(g.mJacobian),bias});
	}
	for (uint iteration = 0; iteration < steps.GetNumVelocitySteps(); ++iteration) {
		bool changed = false;
		for (auto &row : rows) {
			const double k = row.mResponse.mInverseEffectiveMass;
			if (k == 0) continue;
			const double lambda = std::max(0.0,row.mLambda-(mResponse.Velocity(row.mJacobian)+row.mBias)/k);
			const double delta = lambda-row.mLambda; row.mLambda = lambda;
			mResponse.Apply(row.mResponse,delta); changed |= delta != 0;
		}
		if (!changed) break;
	}
	double scale = 1;
	for (size_t i = 0; i < mBodies.size(); ++i) {
		const auto &v = mResponse.GetBody(i).mVelocity;
		for (double value : v) if (!std::isfinite(value) || std::abs(value) > FLT_MAX) return EResult::Invalid;
		const double angle = std::hypot(v[3],v[4],v[5]);
		if (angle > .25) scale = std::min(scale,.25/angle);
	}
	// Backtracking is a single nonlinear projection, using the same displacement
	// for every body. It never switches to a different physical solver.
	for (uint trial = 0; trial < 12; ++trial,scale *= .5) {
		auto poses = mPoses;
		bool moved = false;
		for (size_t i = 0; i < mBodies.size(); ++i) if (mBodies[i]->IsDynamic()) {
			const auto &v = mResponse.GetBody(i).mVelocity;
			poses[i].mPosition += Vec3(float(scale*v[0]),float(scale*v[1]),float(scale*v[2]));
			const Vec3 rotation(float(scale*v[3]),float(scale*v[4]),float(scale*v[5]));
			const float angle = rotation.Length();
			if (angle > 1.0e-6f) poses[i].mRotation = (Quat::sRotation(rotation/angle,angle)*poses[i].mRotation).Normalized();
			moved |= poses[i].mPosition != mPoses[i].mPosition || poses[i].mRotation != mPoses[i].mRotation;
		}
		if (!moved) return EResult::Unchanged;
		std::vector<Geometry> candidate;
		if (!GeometryAt(poses,candidate) || !(Merit(candidate) < initial)) continue;
		for (size_t i = 0; i < mBodies.size(); ++i) if (mBodies[i]->IsDynamic()) {
			const auto &v = mResponse.GetBody(i).mVelocity;
			mBodies[i]->AddPositionStep(Vec3(float(scale*v[0]),float(scale*v[1]),float(scale*v[2])));
			mBodies[i]->AddRotationStep(Vec3(float(scale*v[3]),float(scale*v[4]),float(scale*v[5])));
		}
		return EResult::Applied;
	}
	// Least-squares or finite-precision stagnation is not a claim that all
	// original constraints are satisfied. The caller stops this iteration.
	return EResult::Unchanged;
}

JointPositionSolver::EResult JointPositionSolver::Solve(Constraint **constraints, const uint32 *begin, const uint32 *end,
	ContactConstraintManager &manager, const uint32 *cbegin, const uint32 *cend, const PhysicsSettings &settings)
{
	JointPositionSolver solver(settings);
	return solver.Run(constraints,begin,end,manager,cbegin,cend);
}

JPH_NAMESPACE_END
