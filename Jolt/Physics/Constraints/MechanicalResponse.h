// SPDX-License-Identifier: MIT
#pragma once

#include <Jolt/Jolt.h>
#include <array>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

JPH_NAMESPACE_BEGIN

/// Mass response of mechanically connected bodies. Contacts do not join blocks.
/// A row is J*v+b+compliance*lambda=0. Armature uses b=-rotor_speed,
/// compliance=1/armature; geometric bilateral rows have zero compliance.
/// All factorization and accumulated velocities use double precision.
/// The collector supplies native masked mobility and current geometry for one
/// physics substep. Prepared impulses belong to that fixed body/equation set.
class MechanicalResponse
{
public:
	using Spatial = std::array<double, 6>;
	using SpatialMatrix = std::array<double, 36>;
	struct BodyState
	{
		bool mDynamic = false;
		Spatial mVelocity{};
		SpatialMatrix mInverseMass{};
	};
	struct Term
	{
		size_t mBody = 0;
		Spatial mJacobian{};
	};
	struct Jacobian
	{
		Term mFirst, mSecond;
	};
	struct Equation
	{
		Jacobian mJacobian;
		double mBias = 0.0;
		double mCompliance = 0.0;
	};
	struct Impulse
	{
		struct Contribution
		{
			size_t mBlock;
			std::vector<double> mVelocity;
		};
		std::vector<Contribution> mContributions;
		double mInverseEffectiveMass = 0.0;
	};

	/// A two-row PSD response, represented in principal coordinates. The
	/// eigenvalues are factor norms, including a genuinely singular plane.
	struct Plane
	{
		double mCross = 0, mCos = 1, mSin = 0;
		double mEigen[2]{};

		/// Maximum-dissipation Coulomb disk: min .5*l*K*l + q*l.
		bool SolveDisk(double inQ1, double inQ2, double inRadius, std::array<double,2> &outLambda) const
		{
			if (inRadius <= 0) { outLambda = {}; return true; }
			const double q[2] = {mCos*inQ1+mSin*inQ2,-mSin*inQ1+mCos*inQ2};
			double value[2]{};
			bool free = true;
			for (size_t i = 0; i < 2; ++i) {
				if (mEigen[i] > 0) value[i] = -q[i]/mEigen[i];
				else if (q[i] != 0) free = false;
			}
			if (!free || std::hypot(value[0],value[1]) > inRadius) {
				// The secular equation is decreasing and convex. Newton from
				// this necessary lower bound approaches its root from below,
				// including planes whose two physical scales differ greatly.
				const double scaled[2] = {q[0]/inRadius,q[1]/inRadius};
				double shift = std::max({0.0,std::abs(scaled[0])-mEigen[0],std::abs(scaled[1])-mEigen[1]});
				bool converged = false;
				for (uint iteration = 0; iteration < 64; ++iteration) {
					double ratio[2]{};
					for (size_t i = 0; i < 2; ++i)
						if (scaled[i] != 0) ratio[i] = scaled[i]/(mEigen[i]+shift);
					const double length = std::hypot(ratio[0],ratio[1]);
					if (std::abs(length-1) <= 8*std::numeric_limits<double>::epsilon()) { converged = true; break; }
					double scale = std::numeric_limits<double>::max();
					for (size_t i = 0; i < 2; ++i)
						if (scaled[i] != 0) scale = std::min(scale,mEigen[i]+shift);
					double slope = 0;
					for (size_t i = 0; i < 2; ++i)
						if (ratio[i] != 0) slope += (ratio[i]/length)*ratio[i]*(scale/(mEigen[i]+shift));
					const double next = shift + scale*((length-1)/slope);
					if (next == shift) break;
					shift = next;
				}
				// Failure stays on the owner's existing error path;
				// it must not hang a physics worker or change the physical model.
				if (!converged) return false;
				for (size_t i = 0; i < 2; ++i) value[i] = q[i] == 0? 0 : -q[i]/(mEigen[i]+shift);
			}
			outLambda = {mCos*value[0]-mSin*value[1],mSin*value[0]+mCos*value[1]};
			return true;
		}
	};
	struct ImpulsePair { Impulse mFirst, mSecond; Plane mPlane; };

	size_t AddBody(const BodyState &inBody)
	{
		mBodies.push_back(inBody);
		// A fully locked dynamic body is an affine boundary, not a bridge
		// between otherwise independent mechanisms.
		mBodies.back().mDynamic &= std::any_of(inBody.mInverseMass.begin(), inBody.mInverseMass.end(),
			[](double value) { return value != 0.0; });
		return mBodies.size() - 1;
	}
	void AddEquation(const Equation &inEquation) { mEquations.push_back(inEquation); }
	const BodyState &GetBody(size_t inBody) const { return mBodies[inBody]; }
	double GetMaxHardResidual() const
	{
		double maximum = 0.0;
		for (const auto &equation : mEquations)
			if (equation.mCompliance == 0.0)
				maximum = std::max(maximum, std::abs(Velocity(equation.mJacobian) + equation.mBias));
		return maximum;
	}

	/// Returns false if the responsive physical mass cannot be factored. The owner
	/// also checks the original hard-equation residual before publishing state.
	/// No candidate body velocity has been written to the physics world here.
	[[nodiscard]] bool Build() { return BuildInternal(false); }

	/// Position projection uses the reachable least-squares target for geometric
	/// rows. Nonlinear closed loops may have an inconsistent linearized RHS.
	/// Original equations remain available for residual diagnostics.
	[[nodiscard]] bool BuildPosition() { return BuildInternal(true); }

private:
	bool BuildInternal(bool inProjectHardTargets)
	{
		const size_t count = mBodies.size();
		std::vector<size_t> parent(count);
		std::iota(parent.begin(), parent.end(), size_t(0));
		auto root = [&](size_t body) {
			while (parent[body] != body) {
				parent[body] = parent[parent[body]];
				body = parent[body];
			}
			return body;
		};
		for (const auto &equation : mEquations) {
			const size_t a = equation.mJacobian.mFirst.mBody, b = equation.mJacobian.mSecond.mBody;
			if (mBodies[a].mDynamic && mBodies[b].mDynamic)
				parent[root(b)] = root(a);
		}
		mBlocks.clear();
		mLocations.assign(count, {cNoBlock, 0});
		std::vector<size_t> roots(count, cNoBlock);
		for (size_t body = 0; body < count; ++body) {
			if (!mBodies[body].mDynamic)
				continue;
			const size_t r = root(body);
			if (roots[r] == cNoBlock) {
				roots[r] = mBlocks.size();
				mBlocks.emplace_back();
			}
			auto &block = mBlocks[roots[r]];
			mLocations[body] = {roots[r], block.mBodies.size()};
			block.mBodies.push_back(body);
		}
		for (size_t row = 0; row < mEquations.size(); ++row) {
			const auto &j = mEquations[row].mJacobian;
			const size_t block = mBodies[j.mFirst.mBody].mDynamic? mLocations[j.mFirst.mBody].first
				: mLocations[j.mSecond.mBody].first;
			if (block != cNoBlock)
				mBlocks[block].mEquations.push_back(row);
		}
		for (auto &block : mBlocks)
			if (!BuildBlock(block, inProjectHardTargets))
				return false;
		return true;
	}

public:
	double Velocity(const Jacobian &inJacobian) const
	{
		double result = 0.0;
		for (const auto *term : {&inJacobian.mFirst, &inJacobian.mSecond})
			for (size_t axis = 0; axis < 6; ++axis)
				result += term->mJacobian[axis] * mBodies[term->mBody].mVelocity[axis];
		return result;
	}

	Impulse Prepare(const Jacobian &inJacobian) const { return PrepareInternal(inJacobian,nullptr); }

	ImpulsePair PreparePair(const Jacobian &inFirst, const Jacobian &inSecond) const
	{
		std::vector<FactorCoordinates> first, second;
		ImpulsePair result;
		result.mFirst = PrepareInternal(inFirst,&first);
		result.mSecond = PrepareInternal(inSecond,&second);
		auto &plane = result.mPlane;
		for (const auto &a : first)
			for (const auto &b : second)
				if (a.mBlock == b.mBlock) plane.mCross += Dot(a.mValues,b.mValues);
		if (plane.mCross != 0) {
			const double angle = .5*std::atan2(2*plane.mCross,result.mFirst.mInverseEffectiveMass-result.mSecond.mInverseEffectiveMass);
			plane.mCos = std::cos(angle); plane.mSin = std::sin(angle);
		}
		std::vector<size_t> blocks;
		for (const auto *set : {&first,&second})
			for (const auto &entry : *set)
				if (std::find(blocks.begin(),blocks.end(),entry.mBlock) == blocks.end()) blocks.push_back(entry.mBlock);
		for (size_t block : blocks) {
			const std::vector<double> *a = nullptr, *b = nullptr;
			for (const auto &entry : first) if (entry.mBlock == block) a = &entry.mValues;
			for (const auto &entry : second) if (entry.mBlock == block) b = &entry.mValues;
			const size_t size = a? a->size() : b->size();
			for (size_t i = 0; i < size; ++i) {
				const double x = a? (*a)[i] : 0, y = b? (*b)[i] : 0;
				const double u = plane.mCos*x+plane.mSin*y, v = -plane.mSin*x+plane.mCos*y;
				plane.mEigen[0] += u*u;
				plane.mEigen[1] += v*v;
			}
		}
		return result;
	}

private:
	struct FactorCoordinates { size_t mBlock; std::vector<double> mValues; };
	Impulse PrepareInternal(const Jacobian &inJacobian, std::vector<FactorCoordinates> *outCoordinates) const
	{
		Impulse result;
		for (const auto *term : {&inJacobian.mFirst, &inJacobian.mSecond}) {
			const size_t index = mLocations[term->mBody].first;
			if (index == cNoBlock || std::any_of(result.mContributions.begin(), result.mContributions.end(),
				[index](const auto &entry) { return entry.mBlock == index; }))
				continue;
			const auto &block = mBlocks[index];
			const size_t n = 6 * block.mBodies.size();
			std::vector<double> j(n, 0.0);
			// Combine both endpoints BEFORE multiplying by the block response.
			// Intra-assembly contacts have cross terms between the endpoints.
			for (const auto *endpoint : {&inJacobian.mFirst, &inJacobian.mSecond})
				if (mLocations[endpoint->mBody].first == index)
					for (size_t a = 0; a < 6; ++a)
						j[6 * mLocations[endpoint->mBody].second + a] += endpoint->mJacobian[a];
			std::vector<double> reduced(block.mDimension, 0.0);
			for (size_t a = 0; a < n; ++a)
				for (size_t col = 0; col < block.mDimension; ++col)
					reduced[col] += block.mBasis[a * block.mDimension + col] * j[a];
			double mobility = 0.0;
			FactorCoordinates factor{index,{}};
			const auto solved = block.mMass.Solve(reduced, &mobility, outCoordinates? &factor.mValues : nullptr);
			if (outCoordinates) outCoordinates->push_back(std::move(factor));
			Impulse::Contribution contribution{index, std::vector<double>(n, 0.0)};
			for (size_t a = 0; a < n; ++a)
				for (size_t col = 0; col < block.mDimension; ++col)
					contribution.mVelocity[a] += block.mBasis[a * block.mDimension + col] * solved[col];
			// This is a squared factor norm, not a cancellation of bare
			// and projected mobilities. It cannot manufacture a negative K.
			result.mInverseEffectiveMass += mobility;
			result.mContributions.push_back(std::move(contribution));
		}
		return result;
	}

public:
	void Apply(const Impulse &inImpulse, double inLambda)
	{
		for (const auto &entry : inImpulse.mContributions) {
			const auto &block = mBlocks[entry.mBlock];
			for (size_t body = 0; body < block.mBodies.size(); ++body)
				for (size_t a = 0; a < 6; ++a)
					mBodies[block.mBodies[body]].mVelocity[a] += inLambda * entry.mVelocity[6 * body + a];
		}
	}

private:
	/// Positive definite mass factor. Only HARD geometry gets rank reduction;
	/// every finite armature contributes to this mass without a rank threshold.
	class MassFactor
	{
	public:
		bool Build(const std::vector<double> &inMatrix, size_t inSize)
		{
			mSize = inSize;
			mScale.resize(inSize);
			mLower.assign(inSize * inSize, 0.0);
			for (size_t a = 0; a < inSize; ++a) {
				if (!(inMatrix[a * inSize + a] > 0.0) || !std::isfinite(inMatrix[a * inSize + a]))
					return false;
				mScale[a] = std::sqrt(inMatrix[a * inSize + a]);
			}
			for (size_t a = 0; a < inSize; ++a)
				for (size_t b = 0; b <= a; ++b) {
					double value = inMatrix[a * inSize + b] / mScale[a] / mScale[b];
					for (size_t k = 0; k < b; ++k)
						value -= mLower[a * inSize + k] * mLower[b * inSize + k];
					if (a == b) {
						if (!(value > 0.0) || !std::isfinite(value))
							return false;
						mLower[a * inSize + b] = std::sqrt(value);
					} else
						mLower[a * inSize + b] = value / mLower[b * inSize + b];
				}
			return true;
		}

		std::vector<double> Solve(const std::vector<double> &inRhs, double *outQuadratic = nullptr, std::vector<double> *outForward = nullptr) const
		{
			std::vector<double> values(mSize), result(mSize);
			double quadratic = 0.0;
			for (size_t a = 0; a < mSize; ++a) {
				values[a] = inRhs[a] / mScale[a];
				for (size_t b = 0; b < a; ++b)
					values[a] -= mLower[a * mSize + b] * values[b];
				values[a] /= mLower[a * mSize + a];
				quadratic += values[a] * values[a];
			}
			if (outQuadratic)
				*outQuadratic = quadratic;
			if (outForward) *outForward = values;
			for (size_t a = mSize; a-- > 0;) {
				for (size_t b = a + 1; b < mSize; ++b)
					values[a] -= mLower[b * mSize + a] * values[b];
				values[a] /= mLower[a * mSize + a];
				result[a] = values[a] / mScale[a];
			}
			return result;
		}

	private:
		size_t mSize = 0;
		std::vector<double> mScale, mLower;
	};

	struct Block
	{
		std::vector<size_t> mBodies, mEquations;
		size_t mDimension = 0;
		std::vector<double> mBasis;
		MassFactor mMass;
	};

	static double Dot(const std::vector<double> &a, const std::vector<double> &b)
	{
		return std::inner_product(a.begin(), a.end(), b.begin(), 0.0);
	}

	/// Pivoted twice-orthogonalized row QR, including the affine right hand
	/// side. A dependent hard row remains a hard equation and is checked after
	/// the baseline solve. This never deletes a finite inertia contribution.
	static bool HardBasis(std::vector<std::vector<double>> rows,
		std::vector<double> rhs, size_t inSize, bool inProjectTargets,
		std::vector<std::vector<double>> &q, std::vector<double> &outParticular)
	{
		std::vector<double> targets;
		outParticular.assign(inSize, 0.0);
		// Row normalization conditions the rank calculation only. Position
		// targets minimize the original geometric residual used by acceptance.
		const auto originalRows = inProjectTargets? rows : std::vector<std::vector<double>>{};
		const auto originalRhs = inProjectTargets? rhs : std::vector<double>{};
		for (size_t row = 0; row < rows.size(); ++row) {
			const double norm = std::sqrt(Dot(rows[row], rows[row]));
			if (norm > 0.0) {
				for (double &value : rows[row]) value /= norm;
				rhs[row] /= norm;
			}
		}
		const double tolerance = 16.0 * std::numeric_limits<double>::epsilon() * double(std::max(inSize, rows.size()));
		for (size_t k = 0; k < rows.size(); ++k) {
			size_t pivot = k;
			for (size_t row = k + 1; row < rows.size(); ++row)
				if (Dot(rows[row], rows[row]) > Dot(rows[pivot], rows[pivot])) pivot = row;
			std::swap(rows[k], rows[pivot]);
			std::swap(rhs[k], rhs[pivot]);
			for (size_t pass = 0; pass < 2; ++pass)
				for (size_t previous = 0; previous < q.size(); ++previous) {
					const double coefficient = Dot(rows[k], q[previous]);
					for (size_t axis = 0; axis < inSize; ++axis) rows[k][axis] -= coefficient * q[previous][axis];
					rhs[k] -= coefficient * targets[previous];
				}
			const double norm = std::sqrt(Dot(rows[k], rows[k]));
			if (norm <= tolerance) break;
			for (double &value : rows[k]) value /= norm;
			q.push_back(rows[k]);
			targets.push_back(rhs[k] / norm);
			for (size_t row = k + 1; row < rows.size(); ++row)
				for (size_t pass = 0; pass < 2; ++pass) {
					const double coefficient = Dot(rows[row], q.back());
					for (size_t axis = 0; axis < inSize; ++axis) rows[row][axis] -= coefficient * q.back()[axis];
					rhs[row] -= coefficient * targets.back();
				}
		}
		if (inProjectTargets && !q.empty()) {
			// A*Q^T has full column rank. Factor its columns directly rather
			// than squaring its condition number with normal equations.
			const size_t m = originalRows.size(), rank = q.size();
			std::vector<std::vector<double>> columns(rank, std::vector<double>(m));
			std::vector<double> upper(rank * rank, 0.0), projected(rank);
			for (size_t col = 0; col < rank; ++col) {
				for (size_t row = 0; row < m; ++row)
					columns[col][row] = Dot(originalRows[row], q[col]);
				for (size_t pass = 0; pass < 2; ++pass)
					for (size_t previous = 0; previous < col; ++previous) {
						const double coefficient = Dot(columns[col], columns[previous]);
						upper[previous * rank + col] += coefficient;
						for (size_t row = 0; row < m; ++row) columns[col][row] -= coefficient * columns[previous][row];
					}
				const double norm = std::sqrt(Dot(columns[col], columns[col]));
				if (!(norm > 0.0) || !std::isfinite(norm)) return false;
				upper[col * rank + col] = norm;
				for (double &value : columns[col]) value /= norm;
				projected[col] = Dot(columns[col], originalRhs);
			}
			for (size_t col = rank; col-- > 0;) {
				for (size_t later = col + 1; later < rank; ++later) projected[col] -= upper[col * rank + later] * targets[later];
				targets[col] = projected[col] / upper[col * rank + col];
			}
		}
		for (size_t row = 0; row < q.size(); ++row)
			for (size_t axis = 0; axis < inSize; ++axis) outParticular[axis] += q[row][axis] * targets[row];
		return true;
	}

	static std::vector<std::vector<double>> NullBasis(const std::vector<std::vector<double>> &inHard, size_t inSize)
	{
		std::vector<std::vector<double>> candidates(inSize, std::vector<double>(inSize, 0.0)), basis;
		for (size_t axis = 0; axis < inSize; ++axis) {
			candidates[axis][axis] = 1.0;
			for (size_t pass = 0; pass < 2; ++pass)
				for (const auto &q : inHard) {
					const double coefficient = Dot(candidates[axis], q);
					for (size_t a = 0; a < inSize; ++a) candidates[axis][a] -= coefficient * q[a];
				}
		}
		while (basis.size() + inHard.size() < inSize) {
			auto pivot = std::max_element(candidates.begin(), candidates.end(),
				[](const auto &a, const auto &b) { return Dot(a, a) < Dot(b, b); });
			auto q = *pivot;
			for (size_t pass = 0; pass < 2; ++pass) {
				for (const auto &old : inHard) {
					const double coefficient = Dot(q, old);
					for (size_t a = 0; a < inSize; ++a) q[a] -= coefficient * old[a];
				}
				for (const auto &old : basis) {
					const double coefficient = Dot(q, old);
					for (size_t a = 0; a < inSize; ++a) q[a] -= coefficient * old[a];
				}
			}
			const double norm = std::sqrt(Dot(q, q));
			for (double &value : q) value /= norm;
			basis.push_back(q);
			for (auto &candidate : candidates) {
				const double coefficient = Dot(candidate, q);
				for (size_t a = 0; a < inSize; ++a) candidate[a] -= coefficient * q[a];
			}
		}
		return basis;
	}

	bool BuildBlock(Block &ioBlock, bool inProjectHardTargets)
	{
		const size_t n = 6 * ioBlock.mBodies.size();
		std::vector<double> mass(n * n, 0.0);
		std::vector<size_t> responsive;
		for (size_t body = 0; body < ioBlock.mBodies.size(); ++body) {
			const auto &inverse = mBodies[ioBlock.mBodies[body]].mInverseMass;
			std::vector<size_t> coordinates;
			for (size_t axis = 0; axis < 6; ++axis)
				if (inverse[axis * 6 + axis] > 0.0) {
					coordinates.push_back(axis);
					responsive.push_back(6 * body + axis);
				}
			const size_t count = coordinates.size();
			std::vector<double> local(count * count), rhs(count, 0.0);
			for (size_t a = 0; a < count; ++a)
				for (size_t b = 0; b < count; ++b) local[a * count + b] = inverse[coordinates[a] * 6 + coordinates[b]];
			MassFactor factor;
			if (!factor.Build(local, count))
				return false;
			for (size_t col = 0; col < count; ++col) {
				rhs[col] = 1.0;
				const auto solved = factor.Solve(rhs);
				rhs[col] = 0.0;
				for (size_t row = 0; row < count; ++row)
					mass[(6 * body + coordinates[row]) * n + 6 * body + coordinates[col]] = solved[row];
			}
		}
		const size_t d = responsive.size();
		std::vector<std::vector<double>> jacobians;
		std::vector<std::vector<double>> hard;
		std::vector<double> hardRhs;
		for (size_t index : ioBlock.mEquations) {
			const auto &equation = mEquations[index];
			std::vector<double> j(n, 0.0);
			for (const auto *term : {&equation.mJacobian.mFirst, &equation.mJacobian.mSecond})
				if (mBodies[term->mBody].mDynamic)
					for (size_t a = 0; a < 6; ++a) j[6 * mLocations[term->mBody].second + a] += term->mJacobian[a];
			jacobians.push_back(j);
			if (equation.mCompliance == 0.0) {
				std::vector<double> row(d);
				for (size_t a = 0; a < d; ++a) row[a] = j[responsive[a]];
				hard.push_back(std::move(row));
				hardRhs.push_back(-Velocity(equation.mJacobian) - equation.mBias);
			}
		}
		std::vector<double> localParticular;
		std::vector<std::vector<double>> hardBasis;
		if (!HardBasis(std::move(hard), std::move(hardRhs), d, inProjectHardTargets, hardBasis, localParticular)) return false;
		const auto nullBasis = NullBasis(hardBasis, d);
		const size_t k = ioBlock.mDimension = nullBasis.size();
		std::vector<double> particular(n, 0.0);
		ioBlock.mBasis.assign(n * k, 0.0);
		for (size_t a = 0; a < d; ++a) {
			particular[responsive[a]] = localParticular[a];
			for (size_t col = 0; col < k; ++col) ioBlock.mBasis[responsive[a] * k + col] = nullBasis[col][a];
		}
		std::vector<double> mz(n * k, 0.0), mp(n, 0.0), reduced(k * k, 0.0), rhs(k, 0.0);
		for (size_t a = 0; a < n; ++a)
			for (size_t b = 0; b < n; ++b) {
				mp[a] += mass[a * n + b] * particular[b];
				for (size_t col = 0; col < k; ++col) mz[a * k + col] += mass[a * n + b] * ioBlock.mBasis[b * k + col];
			}
		for (size_t a = 0; a < k; ++a)
			for (size_t axis = 0; axis < n; ++axis) {
				rhs[a] -= ioBlock.mBasis[axis * k + a] * mp[axis];
				for (size_t b = 0; b < k; ++b) reduced[a * k + b] += ioBlock.mBasis[axis * k + a] * mz[axis * k + b];
			}
		for (size_t row = 0; row < ioBlock.mEquations.size(); ++row) {
			const auto &equation = mEquations[ioBlock.mEquations[row]];
			if (equation.mCompliance == 0.0) continue;
			std::vector<double> projection(k, 0.0);
			const auto &j = jacobians[row];
			const double offset = Velocity(equation.mJacobian) + equation.mBias + Dot(j, particular);
			for (size_t a = 0; a < k; ++a)
				for (size_t axis = 0; axis < n; ++axis) projection[a] += j[axis] * ioBlock.mBasis[axis * k + a];
			for (size_t a = 0; a < k; ++a) {
				rhs[a] -= projection[a] * offset / equation.mCompliance;
				for (size_t b = 0; b < k; ++b) reduced[a * k + b] += projection[a] * projection[b] / equation.mCompliance;
			}
		}
		if (!ioBlock.mMass.Build(reduced, k))
			return false;
		const auto solved = ioBlock.mMass.Solve(rhs);
		for (size_t body = 0; body < ioBlock.mBodies.size(); ++body)
			for (size_t a = 0; a < 6; ++a) {
				double delta = particular[6 * body + a];
				for (size_t col = 0; col < k; ++col) delta += ioBlock.mBasis[(6 * body + a) * k + col] * solved[col];
				mBodies[ioBlock.mBodies[body]].mVelocity[a] += delta;
			}
		return true;
	}

	static constexpr size_t cNoBlock = size_t(-1);
	std::vector<BodyState> mBodies;
	std::vector<Equation> mEquations;
	std::vector<Block> mBlocks;
	std::vector<std::pair<size_t, size_t>> mLocations;
};

JPH_NAMESPACE_END
