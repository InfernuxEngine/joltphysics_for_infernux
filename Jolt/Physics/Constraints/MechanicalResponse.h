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
	[[nodiscard]] bool Build()
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
			if (!BuildBlock(block))
				return false;
		return true;
	}

	double Velocity(const Jacobian &inJacobian) const
	{
		double result = 0.0;
		for (const auto *term : {&inJacobian.mFirst, &inJacobian.mSecond})
			for (size_t axis = 0; axis < 6; ++axis)
				result += term->mJacobian[axis] * mBodies[term->mBody].mVelocity[axis];
		return result;
	}

	Impulse Prepare(const Jacobian &inJacobian) const
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
			const auto solved = block.mMass.Solve(reduced, &mobility);
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

		std::vector<double> Solve(const std::vector<double> &inRhs, double *outQuadratic = nullptr) const
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
	static std::vector<std::vector<double>> HardBasis(std::vector<std::vector<double>> rows,
		std::vector<double> rhs, size_t inSize, std::vector<double> &outParticular)
	{
		std::vector<std::vector<double>> q;
		std::vector<double> targets;
		outParticular.assign(inSize, 0.0);
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
		for (size_t row = 0; row < q.size(); ++row)
			for (size_t axis = 0; axis < inSize; ++axis) outParticular[axis] += q[row][axis] * targets[row];
		return q;
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

	bool BuildBlock(Block &ioBlock)
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
		const auto hardBasis = HardBasis(std::move(hard), std::move(hardRhs), d, localParticular);
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
