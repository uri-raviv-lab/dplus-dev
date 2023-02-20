#include "Symmetries.h"


#define _USE_MATH_DEFINES
#include <math.h>
#include "mathfuncs.h"
#include "../backend_version.h"
#include "md5.h"

#include "Grid.h"
#include <time.h>

#ifdef _WIN32
#include <windows.h> // For LoadLibrary
#else
#include <dlfcn.h>
typedef void *HMODULE;
#define GetProcAddress dlsym
#endif

#include "BackendInterface.h"
#include "declarations.h"
#include "UseGPU.h"
#include <GPUHeader.h>

#include "../../Conversions/JsonWriter.h"
#include <rapidjson/writer.h>


void GridSymmetry::OrganizeParameters(const VectorXd& p, int nLayers) {
		Symmetry::OrganizeParameters(p, nLayers);

		size_t nNodeParams = 0;
		const double *nodeparams = ParameterTree::GetNodeParamVec(p.data(), p.size(), nNodeParams, false);

		// Skips nLayers (nodeparams[0])

		da = nodeparams[1];
		db = nodeparams[2];
		dc = nodeparams[3];

		alpha = Radian(Degree(nodeparams[4]));
		while (alpha < -M_PI) alpha = Radian(alpha + 2.0 * M_PI);
		while (alpha >= M_PI) alpha = Radian(alpha - 2.0 * M_PI);
		beta = Radian(Degree(nodeparams[5]));
		while (beta < - M_PI) beta = Radian(beta + 2.0 * M_PI);
		while (beta >= M_PI) beta = Radian(beta - 2.0 * M_PI);
		gamma = Radian(Degree(nodeparams[6]));
		while (gamma < -M_PI) gamma = Radian(gamma + 2.0 * M_PI);
		while (gamma >= M_PI) gamma = Radian(gamma - 2.0 * M_PI);

		Na = nodeparams[7];
		Nb = nodeparams[8];
		Nc = nodeparams[9];

		scale = nodeparams[10];
	}


	void GridSymmetry::PreCalculate(VectorXd& p, int nLayers) {
		Symmetry::PreCalculate(p, nLayers);

		FACC sa, sb, sc, ca, cb, cc, cat;

		printf("alpha = %f\tbeta = %f\tgamma = %f\n", alpha.rad, beta.rad, gamma.rad);

		if (alpha == 0. || beta == 0. || gamma == 0. ||
			fabs(fabs(alpha) - M_PI) < 0.000001 ||
			fabs(fabs(beta) - M_PI) < 0.000001 ||
			fabs(fabs(gamma) - M_PI) < 0.000001
			)
			throw backend_exception(ERROR_INVALIDARGS,
			"Angles in a Space-filling Symmetry cannot be zero or 180."
			" The result is a 2D space. To create a 2D space, make the angle"
			" 90 degrees and one repetition in that direction.");

		if (fabs(alpha) + fabs(beta) <= fabs(gamma) || 
			fabs(gamma) + fabs(beta) <= fabs(alpha) || 
			fabs(alpha) + fabs(gamma) <= fabs(beta))
			throw backend_exception(ERROR_INVALIDARGS,
			"In the Space-filling Symmetry, the sum of any two angles must be greater than the third.");

		sa = sin(alpha);
		sb = sin(beta);
		sc = sin(gamma);
		ca = cos(alpha);
		cb = cos(beta);
		cc = cos(gamma);

		cat = (ca - cb * cc) / (sb * sc);

		// In case something slipped through the cracks...
		if (fabs(cat) > 1.0)
			throw backend_exception(ERROR_INVALIDARGS,
			"In the Space-filling Symmetry, the angles don't form a valid basis.");

		//calculation of Cartesian coordinates
		av = Vector3d(da, 0.0, 0.0);
		bv = Vector3d(db * cc, db * sc, 0.0);
		cv = Vector3d(dc * cb, dc * sb * cat, dc * sb * sin(acos(cat)));

	}


	std::complex<FACC> GridSymmetry::calcAmplitude(FACC qx, FACC qy, FACC qz) {
		// Constants
		const std::complex<FACC> Im(0.0, 1.0), One(1.0, 0.0);

		std::complex<FACC> amp(0.0, 0.0);
		for (int j = 0; j < (int)_amps.size(); j++)
			amp += _amps[j]->getAmplitude(qx, qy, qz);

		// assuming q = alpha * a* + beta * b* +gamma * c* - calculation of alpha, beta, gamma
		double dotA = qx * av[0] + qy * av[1] + qz * av[2];
		double dotB = qx * bv[0] + qy * bv[1] + qz * bv[2];
		double dotC = qx * cv[0] + qy * cv[1] + qz * cv[2];

		//here I need to take to given projection and find the structure factor. Then to multiply calcAmplitude by it.
		// The structure factor is given by a factor dependant of a* times a factor by b* times a factor by c*
		// Each is given by \sum_{n=1}^{Rep_a}\Exp ( i \alpha n )
		if (closeToZero(dotA) || Na == 1.0)
			amp *= Na;
		else
			amp *= (One - exp(Im * dotA * Na)) / (One - exp(Im * dotA));

		if (closeToZero(dotB) || Nb == 1.0)
			amp *= Nb;
		else
			amp *= (One - exp(Im * dotB * Nb)) / (One - exp(Im * dotB));

		if (closeToZero(dotC) || Nc == 1.0)
			amp *= Nc;
		else
			amp *= (One - exp(Im * dotC * Nc)) / (One - exp(Im * dotC));

		return amp * scale;
	}

	std::string GridSymmetry::GetLayerParamName(int index, EDPFunction *edpfunc)
	{
		return GetLayerParamNameStatic(index, edpfunc);
	}
	std::string GridSymmetry::GetLayerParamNameStatic(int index, EDPFunction *edpfunc)
	{
		switch (index) {
		default:
			return Symmetry::GetLayerParamName(index, edpfunc);
		case 0:
			return "Distance";
		case 1:
			return "Angle";
		case 2:
			return "Repetitions";
		}

	}

	std::string GridSymmetry::GetLayerName(int layer)
	{
		return GetLayerNameStatic(layer);
	}
	std::string GridSymmetry::GetLayerNameStatic(int layer)
	{
		if (layer < 0)
			return "WTF?!?!?";
		if (layer > 2)
			return "N/A";

		std::stringstream ss;

		ss << "Vector " << layer + 1;
		return ss.str();
	}

	ExtraParam GridSymmetry::GetExtraParameter(int index)
	{
		return GetExtraParameterStatic(index);
	}

	ExtraParam GridSymmetry::GetExtraParameterStatic(int index)
	{
		if (index == 0) {
			return ExtraParam("Scale", 1.0);
		}
		return ExtraParam();
	}

	double GridSymmetry::GetDefaultParamValue(int paramIndex, int layer, EDPFunction *edpfunc)
	{
		switch (paramIndex) {
		case 0:
			return 2.5;
		case 1:
			return 90.0;
		case 2:
			return 2.0;
		default:
			return Symmetry::GetDefaultParamValue(paramIndex, layer, edpfunc);

		}
	}

	bool GridSymmetry::IsParamApplicable(int layer, int lpindex)
	{
		if (layer >= 0 && layer < 3 &&
			lpindex >= 0 && lpindex < 3) {
			return true;
		}
		return false;
	}

	std::string GridSymmetry::GetDisplayParamName(int index)
	{
		return "N/A";
	}

	double GridSymmetry::GetDisplayParamValue(int index, const paramStruct *p)
	{
		return 0.0;
	}

	void GridSymmetry::GetHeader(unsigned int depth, JsonWriter &writer)
	{

		if (depth == 0)
		{
			writer.Key("Program revision");
			writer.String(BACKEND_VERSION);
		}

		writer.Key("Title");
		writer.String("Space Filling Symmetry");


		writer.Key("Position");
		writer.StartArray();
		writer.Double(tx);
		writer.Double(ty);
		writer.Double(tz);
		writer.EndArray();

		writer.Key("Rotation");
		writer.StartArray();
		writer.Double(ra);
		writer.Double(rb);
		writer.Double(rg);
		writer.EndArray();

		writer.Key("Alpha");
		writer.Double(alpha);
		writer.Key("Beta");
		writer.Double(beta);
		writer.Key("Gamma");
		writer.Double(gamma);

		writer.Key("Distance a");
		writer.Double(da);
		writer.Key("Distance b");
		writer.Double(db);
		writer.Key("Distance c");
		writer.Double(dc);

		writer.Key("Repeats a");
		writer.Double(Na);
		writer.Key("Repeats b");
		writer.Double(Nb);
		writer.Key("Repeats c");
		writer.Double(Nc);

		writer.Key("Scale");
		writer.Double(scale);

		writer.Key("Used Grid");
		writer.Bool(bUseGrid);

		writer.Key("SubModels");
		writer.StartArray();
		for (int i = 0; i < _amps.size(); i++) {
			_amps[i]->GetHeader(depth + 1, writer);
		}
		writer.EndArray();
	}

	void GridSymmetry::GetHeader(unsigned int depth, std::string &header) {
		std::string ampers;
		ampers.resize(depth + 1, '#');
		ampers.append(" ");

		std::stringstream ss;

		if (depth == 0) {
			header.append(ampers + "Program revision: " + BACKEND_VERSION + "\n");
		}
		header.append(ampers + "//////////////////////////////////////\n");

		ss << "Space filling symmetry\n";
		header.append(ampers + ss.str());
		ss.str("");
		ss << "Position (" << tx << "," << ty << "," << tz << ")\n";
		header.append(ampers + ss.str());
		ss.str("");
		ss << "Rotation (" << ra << "," << rb << "," << rg << ")\n";
		header.append(ampers + ss.str());
		ss.str("");

		ss << "Alpha " << alpha << "\n";
		header.append(ampers + ss.str());
		ss.str("");
		ss << "Beta " << beta << "\n";
		header.append(ampers + ss.str());
		ss.str("");
		ss << "Gamma " << gamma << "\n";
		header.append(ampers + ss.str());
		ss.str("");

		ss << "Distance a " << da << "\n";
		header.append(ampers + ss.str());
		ss.str("");
		ss << "Distance b " << db << "\n";
		header.append(ampers + ss.str());
		ss.str("");
		ss << "Distance c " << dc << "\n";
		header.append(ampers + ss.str());
		ss.str("");

		ss << "Repeats a " << Na << "\n";
		header.append(ampers + ss.str());
		ss.str("");
		ss << "Repeats b " << Nb << "\n";
		header.append(ampers + ss.str());
		ss.str("");
		ss << "Repeats c " << Nc << "\n";
		header.append(ampers + ss.str());
		ss.str("");
		ss << "Scale: " << this->scale << "\n";
		header.append(ampers + ss.str());

		ss.str("");
		ss << "Used grid: " << this->bUseGrid << "\n";
		header.append(ampers + ss.str());

		for (int i = 0; i < _amps.size(); i++) {
			_amps[i]->GetHeader(depth + 1, header);
		}

	}

	void GridSymmetry::calculateGrid(FACC qMax, int sections /*= 150*/, progressFunc progFunc /*= NULL*/, void *progArgs /*= NULL*/, double progMin /*= 0.0*/, double progMax /*= 1.0*/, int *pStop /*= NULL */) {
		if (gridStatus == AMP_CACHED) {
			if (PDB_OK == ReadAmplitudeFromCache())
				return;
		}
		clock_t bg = clock(), nd1;
		if (g_useGPUAndAvailable) {
			gpuCalcSpcFllSym = (GPUCalculateSpcFllSymmetry_t)GPUCalcSpcFillSymJacobSphrDD;//GetProcAddress((HMODULE)g_gpuModule, "GPUCalcSpcFillSymJacobSphrDF");
		}

		// Hybrid
		if (!bUseGrid) {
			PDB_READER_ERRS res = CalculateSubAmplitudeGrids(qMax, sections, progFunc, progArgs, progMin, (progMin + progMax) * 0.5, pStop);
			return;
		}

		//	bUseGPU = false;
		// TODO::GPU Another check should be that they all useGrid and have the same size
		if (!bUseGPU || !g_useGPUAndAvailable || (gpuCalcSym == NULL && gpuCalcSpcFllSym == NULL)) {
			Symmetry::calculateGrid(qMax, sections, progFunc, progArgs, progMin, progMax, pStop);
			return;
		}

		PDB_READER_ERRS res = CalculateSubAmplitudeGrids(qMax, sections, progFunc, progArgs, progMin, (progMin + progMax) * 0.5, pStop);
		if (res != PDB_OK) {
			status = res;
			return;
		}
		nd1 = clock();

		PDB_READER_ERRS dbg = getError();
		if (!bUseGrid) {
			std::cout << "Not using grid";
			return;
		}

		InitializeGrid(qMax, sections);
		memset(grid->GetDataPointer(), 0, grid->GetRealSize());

		long long voxels = grid->GetRealSize() / (sizeof(double) * 2);

		// Prepare the parameters for the GPU
		std::vector<FACC> unitCellVectors(9), aveReps(3), iTr(3);
		std::vector<FACC>  iRt(3);

#pragma unroll 3
		for (int i = 0; i < 3; i++) {
			unitCellVectors[i] = av[i];
			unitCellVectors[3 + i] = bv[i];
			unitCellVectors[6 + i] = cv[i];
		}
		aveReps[0] = Na; aveReps[1] = Nb; aveReps[2] = Nc;

		const int thetaDivs = grid->GetDimY(1) - 1;
		const int phiDivs = grid->GetDimZ(1, 1);

		int gpuRes = 0;

		for (int i = 0; i < _amps.size() && gpuRes == 0; i++) {
			JacobianSphereGrid *JGrid = (JacobianSphereGrid*)(_amps[i]->GetInternalGridPointer());
#ifdef _DEBUG
			JGrid->DebugMethod();
#endif
			_amps[i]->GetTranslationRotationVariables(iTr[0], iTr[1], iTr[2],
				iRt[0], iRt[1], iRt[2]);

			gpuRes = gpuCalcSpcFllSym(voxels, thetaDivs, phiDivs, grid->GetStepSize(),
				_amps[i]->GetDataPointer(), grid->GetDataPointer(), JGrid->GetInterpolantPointer(),
				&unitCellVectors[0], &aveReps[0], iRt.data(), &iTr[0], scale,
				progFunc, progArgs, (progMin + progMax) * 0.5, progMax, pStop);
		}
		grid->RunAfterReadingCache(); // CalculateSplines

		if (gpuRes != 0) {
			std::cout << "Error in kernel: " << gpuRes << ". Starting CPU calculations." << std::endl;
			Symmetry::calculateGrid(qMax, sections, progFunc, progArgs, (progMin + progMax) * 0.5, progMax, pStop);
			return;
		}

		for (int i = 0; i < _amps.size(); i++)
			_amps[i]->WriteAmplitudeToCache();

		gridStatus = grid->Validate() ? AMP_READY : AMP_HAS_INVALID_NUMBER;
	}

	std::string GridSymmetry::Hash() const
	{
		std::string str = BACKEND_VERSION "Grid Symmetry";

		str += std::to_string(alpha) + std::to_string(beta) + std::to_string(gamma); // Angle
		str += std::to_string(da) + std::to_string(db) + std::to_string(dc); // Distance
		str += std::to_string(Na) + std::to_string(Nb) + std::to_string(Nc); // Average repetitions

		for (const auto& child : _amps)
			str += child->Hash();

		return md5(str);
	}

	std::string GridSymmetry::GetName() const {
		return "Grid Symmetry";
	}

	bool GridSymmetry::Populate(const VectorXd& p, int nLayers) {
		VectorXd tmp = p;
		PreCalculate(tmp, nLayers);
		return true;
	}

	unsigned int GridSymmetry::GetNumSubLocations() {
		return (unsigned int)(Na * Nb * Nc); // Should this be multiplied by the number of subamplitudes as in ManualSymmetry (line 712)?
	}

	LocationRotation GridSymmetry::GetSubLocation(int posindex) {
		if (posindex < 0 || posindex >= GetNumSubLocations())
			return LocationRotation();

		int i, j, k;

		k = posindex % (int)Nc;
		j = (posindex / (int)Nc) % (int)Nb;
		i = ((posindex / (int)Nc) / (int)Nb) % (int)Na;
		// This is problematic - notice the y and z coordinates changes even if there is only 
		//return LocationRotation(double(i) * av.x() + double(j) * av.y() + double(k) * av.z(),
		//	double(i) * bv.x() + double(j) * bv.y() + double(k) * bv.z(),
		//	double(i) * cv.x() + double(j) * cv.y() + double(k) * cv.z() );
		return LocationRotation(double(i) * av.x() + double(j) * bv.x() + double(k) * cv.x(),
			double(i) * av.y() + double(j) * bv.y() + double(k) * cv.y(),
			double(i) * av.z() + double(j) * bv.z() + double(k) * cv.z());
	}

	ArrayXcX GridSymmetry::getAmplitudesAtPoints(const std::vector<FACC> & relevantQs, FACC theta, FACC phi)
	{
		// First, take orientation of object into account, i.e. change theta and phi to newTheta and newPhi
		FACC st = sin(theta);
		FACC ct = cos(theta);
		FACC sp = sin(phi);
		FACC cp = cos(phi);
		const auto Q = relevantQs[0];

		Eigen::Vector3d Qcart(Q * st*cp, Q * st * sp, Q*ct), Qt;
		Eigen::Matrix3d rot;
		Eigen::Vector3d R(tx, ty, tz);
		Qt = (Qcart.transpose() * RotMat) / Q;

		double newTheta = acos(Qt.z());
		double newPhi = atan2(Qt.y(), Qt.x());

		if (newPhi < 0.0)
			newPhi += M_PI * 2.;

		ArrayXcX phases = (
			std::complex<FACC>(0., 1.) *
			(Qt.dot(R) *
			Eigen::Map<const Eigen::ArrayXd>(relevantQs.data(), relevantQs.size()))
			).exp();

		ArrayXcX reses(relevantQs.size());
		reses.setZero();
		if (GetUseGridWithChildren())
		{
			JacobianSphereGrid* jgrid = dynamic_cast<JacobianSphereGrid*>(grid);

			if (jgrid)
			{
				// Get amplitudes from grid
				return jgrid->getAmplitudesAtPoints(relevantQs, newTheta, newPhi) * phases;
			}
		}

		if (GetUseGridAnyChildren())
		{
			const std::complex<FACC> Im(0.0, 1.0), One(1.0, 0.0);

			ArrayXcX results(relevantQs.size());
			results.setZero();

			for (auto& child : _amps)
				results += child->getAmplitudesAtPoints(relevantQs, newTheta, newPhi);

			st = sin(newTheta);
			ct = cos(newTheta);
			sp = sin(newPhi);
			cp = cos(newPhi);

			for (int i = 0; i < relevantQs.size(); i++)
			{
				Eigen::Vector3d qVector(
					relevantQs[i] * st * cp,
					relevantQs[i] * st * sp,
					relevantQs[i] * ct);
				// assuming q = alpha * a* + beta * b* +gamma * c* - calculation of alpha, beta, gamma
				double dotA = qVector.dot(av);  //qx * av[0] + qy * av[1] + qz * av[2];
				double dotB = qVector.dot(bv);  //qx * bv[0] + qy * bv[1] + qz * bv[2];
				double dotC = qVector.dot(cv);  //qx * cv[0] + qy * cv[1] + qz * cv[2];

				//here I need to take to given projection and find the structure factor. Then to multiply calcAmplitude by it.
				// The structure factor is given by a factor dependant of a* times a factor by b* times a factor by c*
				// Each is given by \sum_{n=1}^{Rep_a}\Exp ( i \alpha n )
				if (closeToZero(dotA) || Na == 1.0)
					results[i] *= Na;
				else
					results[i] *= (One - exp(Im * dotA * Na)) / (One - exp(Im * dotA));

				if (closeToZero(dotB) || Nb == 1.0)
					results[i] *= Nb;
				else
					results[i] *= (One - exp(Im * dotB * Nb)) / (One - exp(Im * dotB));

				if (closeToZero(dotC) || Nc == 1.0)
					results[i] *= Nc;
				else
					results[i] *= (One - exp(Im * dotC * Nc)) / (One - exp(Im * dotC));
			} // for i < relevantQs.size()

			return results * scale;

		} // if GetUseGridAnyChildren


		return scale * getAmplitudesAtPointsWithoutGrid(newTheta, newPhi, relevantQs, phases);
	}


	std::complex<FACC> GridSymmetry::getAmplitudeAtPoint(FACC q, FACC theta, FACC phi)
	{
		// First, take orientation of object into account, i.e. change theta and phi to newTheta and newPhi
		FACC st = sin(theta);
		FACC ct = cos(theta);
		FACC sp = sin(phi);
		FACC cp = cos(phi);

		Eigen::Vector3d Qcart(q * st * cp, q * st * sp, q * ct), Qt;
		Eigen::Matrix3d rot;
		Eigen::Vector3d R(tx, ty, tz);
		Qt = (Qcart.transpose() * RotMat) / q;

		double newTheta = acos(Qt.z());
		double newPhi = atan2(Qt.y(), Qt.x());

		if (newPhi < 0.0)
			newPhi += M_PI * 2.;

		std::complex<FACC> phases = exp(std::complex<FACC>(0., 1.) * (Qt.dot(R) * q));

		if (GetUseGridWithChildren())
		{
			JacobianSphereGrid* jgrid = dynamic_cast<JacobianSphereGrid*>(grid);

			if (jgrid)
			{
				// Get amplitudes from grid
				return jgrid->getAmplitudeAtPoint(q, newTheta, newPhi) * phases;
			}
		}

		if (GetUseGridAnyChildren())
		{
			const std::complex<FACC> Im(0.0, 1.0), One(1.0, 0.0);

			std::complex<FACC> result = 0;

			for (auto& child : _amps)
				result += child->getAmplitudeAtPoint(q, newTheta, newPhi);

			st = sin(newTheta);
			ct = cos(newTheta);
			sp = sin(newPhi);
			cp = cos(newPhi);


			Eigen::Vector3d qVector(
				q * st * cp,
				q * st * sp,
				q * ct);
			// assuming q = alpha * a* + beta * b* +gamma * c* - calculation of alpha, beta, gamma
			double dotA = qVector.dot(av);  //qx * av[0] + qy * av[1] + qz * av[2];
			double dotB = qVector.dot(bv);  //qx * bv[0] + qy * bv[1] + qz * bv[2];
			double dotC = qVector.dot(cv);  //qx * cv[0] + qy * cv[1] + qz * cv[2];

			//here I need to take to given projection and find the structure factor. Then to multiply calcAmplitude by it.
			// The structure factor is given by a factor dependant of a* times a factor by b* times a factor by c*
			// Each is given by \sum_{n=1}^{Rep_a}\Exp ( i \alpha n )
			if (closeToZero(dotA) || Na == 1.0)
				result *= Na;
			else
				result *= (One - exp(Im * dotA * Na)) / (One - exp(Im * dotA));

			if (closeToZero(dotB) || Nb == 1.0)
				result *= Nb;
			else
				result *= (One - exp(Im * dotB * Nb)) / (One - exp(Im * dotB));

			if (closeToZero(dotC) || Nc == 1.0)
				result *= Nc;
			else
				result *= (One - exp(Im * dotC * Nc)) / (One - exp(Im * dotC));

			return result * scale;

		} // if GetUseGridAnyChildren

		return scale * getAmplitudeAtPointWithoutGrid(newTheta, newPhi, q, phases);
	}

	//////////////////////////////////////////////////////////////////////////

	static Eigen::Matrix3d EulerDMat(Radian theta, Radian phi, Radian psi) {
		FACC ax, ay, az, c1, c2, c3, s1, s2, s3;
		ax = theta;
		ay = phi;
		az = psi;
		c1 = cos(ax); s1 = sin(ax);
		c2 = cos(ay); s2 = sin(ay);
		c3 = cos(az); s3 = sin(az);
		Eigen::Matrix3d rot;
		rot << c2*c3, -c2*s3, s2,
			c1*s3 + c3*s1*s2, c1*c3 - s1*s2*s3, -c2*s1,
			s1*s3 - c1*c3*s2, c3*s1 + c1*s2*s3, c1*c2;
		return rot;
	}

	void ManualSymmetry::PreCalculate(VectorXd& p, int nLayers) {
		Symmetry::PreCalculate(p, nLayers);


		translationsPerOrientation.clear();

		// Get the actual parameters
		size_t sz = 0;
		const double *actualP = ParameterTree::GetNodeParamVec(p.data(), p.size(), sz, false);

		// Skip nLayers
		actualP++;

		trans.resize(nLayers);
		rot.resize(nLayers);
		rotVars.resize(nLayers);

		for (int i = 0; i < nLayers; i++) {
			trans[i] = Vector3d(actualP[0 * nLayers + i], actualP[1 * nLayers + i], actualP[2 * nLayers + i]);
			rotVars[i] = Vector3d(actualP[3 * nLayers + i], actualP[4 * nLayers + i], actualP[5 * nLayers + i]) * M_PI / 180.0;
			rot[i] = EulerDMat(Radian(rotVars[i](0)), Radian(rotVars[i](1)), Radian(rotVars[i](2)));
			translationsPerOrientation[rot[i]].push_back(trans[i]);
		}

		scale = actualP[6 * nLayers];
	}

	std::complex<FACC> ManualSymmetry::calcAmplitude(FACC qx, FACC qy, FACC qz) {
		int rows = trans.size();
		Vector3d Q(qx, qy, qz);

		const std::complex<FACC> Im(0.0, 1.0);

		std::complex<FACC> result(0.0, 0.0);
		Vector3d qnew;

		for (int i = 0; i < rows; i++) {
			std::complex<double> res_i(0.0, 0.0);
			qnew = Vector3d(Q.transpose() * rot[i]);
			for (int subAmpNo = 0; subAmpNo < _amps.size(); subAmpNo++)
				res_i += _amps[subAmpNo]->getAmplitude(qnew.x(), qnew.y(), qnew.z());

			assert(res_i == res_i);
			result += res_i * exp(Im * (Q.dot(trans[i])));
		}
		return result * scale;
	}

	std::string ManualSymmetry::GetLayerParamName(int index, EDPFunction *edpfunc) {
		return GetLayerParamNameStatic(index, edpfunc);
	}
	std::string ManualSymmetry::GetLayerParamNameStatic(int index, EDPFunction *edpfunc) {
		switch (index) {
		default:
			return "N/A";
		case 0:
			return "X";
		case 1:
			return "Y";
		case 2:
			return "Z";
		case 3:
			return "Alpha";
		case 4:
			return "Beta";
		case 5:
			return "Gamma";
		}
	}

	std::string ManualSymmetry::GetLayerName(int layer) {
		return GetLayerNameStatic(layer);
	}
	std::string ManualSymmetry::GetLayerNameStatic(int layer) {
		return "Instance %d"; // This was originally layer+1, do we have a problem here?
	}

	double ManualSymmetry::GetDefaultParamValue(int paramIndex, int layer, EDPFunction *edpfunc) {
		return 0.0;
	}

	void ManualSymmetry::GetHeader(unsigned int depth, JsonWriter &writer)
	{
		if (depth == 0)
		{
			writer.Key("Program revision");
			writer.String(BACKEND_VERSION);
		}

		writer.Key("Title");
		writer.String("Manual Symmetry");


		writer.Key("Position");
		writer.StartArray();
		writer.Double(tx);
		writer.Double(ty);
		writer.Double(tz);
		writer.EndArray();

		writer.Key("Rotation");
		writer.StartArray();
		writer.Double(ra);
		writer.Double(rb);
		writer.Double(rg);
		writer.EndArray();

		writer.Key("Instances");
		writer.StartArray();
		for (int i = 0; i < trans.size(); i++) {
			writer.StartObject();
			writer.Key("Translation");
			writer.StartArray();
			writer.Double(trans[i].x());
			writer.Double(trans[i].y());
			writer.Double(trans[i].z());
			writer.EndArray();
			writer.Key("Rotation");
			writer.StartArray();
			writer.Double(rotVars[i].x());
			writer.Double(rotVars[i].y());
			writer.Double(rotVars[i].z());
			writer.EndArray();
			writer.EndObject();
		}
		writer.EndArray();

		writer.Key("Scale");
		writer.Double(scale);

		writer.Key("Used Grid");
		writer.Bool(bUseGrid);

		writer.Key("SubModels");
		writer.StartArray();
		for (int i = 0; i < _amps.size(); i++) {
			writer.StartObject();
			_amps[i]->GetHeader(depth + 1, writer);
			writer.EndObject();
		}
		writer.EndArray();
	}
	void ManualSymmetry::GetHeader(unsigned int depth, std::string &header) {
		std::string ampers;
		ampers.resize(depth + 1, '#');
		ampers.append(" ");

		std::stringstream ss;

		if (depth == 0) {
			header.append(ampers + "Program revision: " + BACKEND_VERSION + "\n");
		}
		header.append(ampers + "//////////////////////////////////////\n");

		ss << "Manual symmetry\n";
		header.append(ampers + ss.str());
		ss.str("");
		ss << "Position (" << tx << "," << ty << "," << tz << ")\n";
		header.append(ampers + ss.str());
		ss.str("");
		ss << "Rotation (" << ra << "," << rb << "," << rg << ")\n";
		header.append(ampers + ss.str());
		ss.str("");

		for (int i = 0; i < trans.size(); i++) {
			ss << "Instance " << i << " translation: (" << trans[i].x() << "," << trans[i].y() << "," << trans[i].z() << ")\n";
			header.append(ampers + ss.str());
			ss.str("");
			ss << "Instance " << i << " rotation: (" << rotVars[i].x() << "," << rotVars[i].y() << "," << rotVars[i].z() << ")\n";
			header.append(ampers + ss.str());
			ss.str("");

		}
		ss << "Scale: " << this->scale << "\n";
		header.append(ampers + ss.str());

		ss.str("");
		ss << "Used grid: " << this->bUseGrid << "\n";
		header.append(ampers + ss.str());

		for (int i = 0; i < _amps.size(); i++) {
			_amps[i]->GetHeader(depth + 1, header);
		}

	}
	ExtraParam ManualSymmetry::GetExtraParameter(int index) {
		return GetExtraParameterStatic(index);
	}
	ExtraParam ManualSymmetry::GetExtraParameterStatic(int index) {
		if (index == 0) {
			return ExtraParam("Scale", 1.0);
		}
		return ExtraParam();
	}

	std::string ManualSymmetry::Hash() const
	{
		std::string str = BACKEND_VERSION "Manual Symmetry";
		for (const auto& t : trans)
			str += std::to_string(t.x()) + std::to_string(t.y()) + std::to_string(t.z());
		for (const auto& r : rotVars)
			str += std::to_string(r.x()) + std::to_string(r.y()) + std::to_string(r.z());

		for (const auto& child : _amps)
			str += child->Hash();

		return md5(str);
	}

	void ManualSymmetry::calculateGrid(FACC qMax, int sections /*= 150*/, progressFunc progFunc /*= NULL*/, void *progArgs /*= NULL*/, double progMin /*= 0.0*/, double progMax /*= 1.0*/, int *pStop /*= NULL*/) {
		if (gridStatus == AMP_CACHED) {
			if (PDB_OK == ReadAmplitudeFromCache())
				return;
		}
		//gpuCalcManSym

		if (g_useGPUAndAvailable) {
			// double, double
			gpuCalcManSym = (GPUCalculateManSymmetry_t)GPUCalcManSymJacobSphrDD;// GetProcAddress((HMODULE)g_gpuModule, "GPUCalcManSymJacobSphrDF");
		}

		// Hybrid
		if (!bUseGrid) {
			PDB_READER_ERRS res = CalculateSubAmplitudeGrids(qMax, sections, progFunc, progArgs, progMin, (progMin + progMax) * 0.5, pStop);
			return;
		}

		if (!bUseGPU || !g_useGPUAndAvailable || (gpuCalcManSym == NULL)) {
			Symmetry::calculateGrid(qMax, sections, progFunc, progArgs, progMin, progMax, pStop);
			return;
		}

		PDB_READER_ERRS res = CalculateSubAmplitudeGrids(qMax, sections, progFunc, progArgs, progMin, (progMin + progMax) * 0.5, pStop);

		if (res != PDB_OK) {
			status = res;
			return;
		}

		if (!bUseGrid) {
			std::cout << "Not using grid";
			return;
		}

		InitializeGrid(qMax, sections);
		memset(grid->GetDataPointer(), 0, grid->GetRealSize());

		int copies = trans.size() + 1;	// Plus one for the inner object(s)
		std::vector<double> xx(copies), yy(copies), zz(copies);
		std::vector<double> aa(copies), bb(copies), cc(copies);
		for (int i = 1; i < copies; i++) {
			xx[i] = trans[i - 1](0);
			yy[i] = trans[i - 1](1);
			zz[i] = trans[i - 1](2);
			aa[i] = Radian(rotVars[i - 1](0));	// Already in radians
			bb[i] = Radian(rotVars[i - 1](1));	// Already in radians
			cc[i] = Radian(rotVars[i - 1](2));	// Already in radians
		}
		copies--;	// Minus one so that the inner object isn't counted twice

		long long voxels = grid->GetRealSize() / (sizeof(double) * 2);
		const int thetaDivs = grid->GetDimY(1) - 1;
		const int phiDivs = grid->GetDimZ(1, 1);

		int gpuRes = 0;

		for (int i = 0; i < _amps.size() && gpuRes == 0; i++) {
			JacobianSphereGrid *JGrid = (JacobianSphereGrid*)(_amps[i]->GetInternalGridPointer());
#ifdef _DEBUG
			JGrid->DebugMethod();
#endif
			_amps[i]->GetTranslationRotationVariables(xx[0], yy[0], zz[0], aa[0], bb[0], cc[0]);

			gpuRes = gpuCalcManSym(voxels, thetaDivs, phiDivs, copies, grid->GetStepSize(),
				_amps[i]->GetDataPointer(), grid->GetDataPointer(), JGrid->GetInterpolantPointer(),
				&xx[0], &yy[0], &zz[0], aa.data(), bb.data(), cc.data(), scale, progFunc, progArgs, (progMin + progMax) * 0.5, progMax, pStop);
		}
		grid->RunAfterReadingCache(); // CalculateSplines

		if (gpuRes != 0) {
			std::cout << "Error in kernel: " << gpuRes << ". Starting CPU calculations." << std::endl;
			Symmetry::calculateGrid(qMax, sections, progFunc, progArgs, (progMin + progMax) * 0.5, progMax, pStop);
			return;
		}

		for (int i = 0; i < _amps.size(); i++)
			_amps[i]->WriteAmplitudeToCache();

		gridStatus = grid->Validate() ? AMP_READY : AMP_HAS_INVALID_NUMBER;



	}

	std::string ManualSymmetry::GetName() const {
		return "Manual Symmetry";
	}

	bool ManualSymmetry::Populate(const VectorXd& p, int nLayers) {
		VectorXd tmp = p;
		PreCalculate(tmp, nLayers);
		return true;
	}

	unsigned int ManualSymmetry::GetNumSubLocations() {
		return (unsigned int)trans.size();// * GetNumSubAmplitudes(); // Should this not be multiplied by the number of subamplitudes as in GridSymmetry (line 429)?
	}

	LocationRotation ManualSymmetry::GetSubLocation(int posindex) {
		if (posindex < 0 || posindex >= GetNumSubLocations())
			return LocationRotation();

		return LocationRotation(trans[posindex].x(), trans[posindex].y(), trans[posindex].z(),
			rotVars[posindex].x(), rotVars[posindex].y(), rotVars[posindex].z());
	}

	bool ManualSymmetry::CalculateGridGPU(GridWorkspace& workspace) {
		if (!g_useGPUAndAvailable)
			return false;

		bool res = true;

		if (!gpuManSymmetryHybridAmplitude)
			gpuManSymmetryHybridAmplitude = (GPUHybridManSymmetryAmplitude_t)GPUHybrid_ManSymmetryAmplitudeDLL;// GetProcAddress((HMODULE)g_gpuModule, "GPUHybrid_ManSymmetryAmplitudeDLL");
		if (!gpuManSymmetrySemiHybridAmplitude)
			gpuManSymmetrySemiHybridAmplitude = (GPUSemiHybridManSymmetryAmplitude_t)GPUSemiHybrid_ManSymmetryAmplitudeDLL;// GetProcAddress((HMODULE)g_gpuModule, "GPUSemiHybrid_ManSymmetryAmplitudeDLL");

		for (int i = 0; i < _amps.size(); i++) {
			IGPUGridCalculable *tst = dynamic_cast<IGPUGridCalculable*>(_amps[i]);

			if (tst && tst->ImplementedHybridGPU())	// Calculate directly on the GPU
			{
				if (!gpuManSymmetryHybridAmplitude)
				{
					std::cout << "GPUHybrid_ManSymmetryAmplitudeDLL not loaded. Skipping model." << std::endl;
					res = false;
					continue;
				}
				workspace.calculator->SetNumChildren(workspace, 1);
				tst->SetModel(workspace.children[0]);
				workspace.children[0].computeStream = workspace.computeStream;
				tst->CalculateGridGPU(workspace.children[0]);

				double3 tr, rt;
				_amps[i]->GetTranslationRotationVariables(tr.x, tr.y, tr.z, rt.x, rt.y, rt.z);
				// TODO:: Deal with rotLoc

				res &= gpuManSymmetryHybridAmplitude(workspace, workspace.children[0], f4(tr), f4(rt));

				workspace.calculator->FreeWorkspace(workspace.children[0]);
			}
			else	// Calculate using CalculateGrid and add to workspace.d_amp
			{
				if (!gpuManSymmetrySemiHybridAmplitude)
				{
					std::cout << "GPUSemiHybrid_ManSymmetryAmplitudeDLL not loaded. Skipping model." << std::endl;
					res = false;
					continue;
				}

				_amps[i]->calculateGrid(workspace.qMax, 2 * (workspace.qLayers - 4));

				double3 tr, rt;
				_amps[i]->GetTranslationRotationVariables(tr.x, tr.y, tr.z, rt.x, rt.y, rt.z);
				// TODO:: Deal with rotLoc
				gpuManSymmetrySemiHybridAmplitude(workspace, _amps[i]->GetDataPointer(), f4(tr), f4(rt));
			}	//  if else tst
		}	// for i

		workspace.calculator->SetNumChildren(workspace, 0);

		float4 dummy;
		dummy.x = 0.;
		dummy.y = 0.;
		dummy.z = 0.;

		// Calculate splines
		if (gpuManSymmetrySemiHybridAmplitude)
			gpuManSymmetrySemiHybridAmplitude(workspace, NULL, dummy, dummy);

		return res;
	}

	bool ManualSymmetry::SetModel(GridWorkspace& workspace) {
		if (!g_useGPUAndAvailable)
			return false;

		if (!gpuHybridSetSymmetry)
			gpuHybridSetSymmetry = (GPUHybridSetSymmetries_t)GPUHybrid_SetSymmetryDLL;// GetProcAddress((HMODULE)g_gpuModule, "GPUHybrid_SetSymmetryDLL");
		if (!gpuHybridSetSymmetry)
			return false;

		std::vector<float4> locs(rotVars.size()), rots(rotVars.size());

		for (int i = 0; i < rotVars.size(); i++) {
			locs[i] = f4(trans[i]);
			rots[i] = f4(rotVars[i]);
		}

		workspace.scale = scale;

		return gpuHybridSetSymmetry(workspace, locs.data(), rots.data(), rots.size());

	}

	bool ManualSymmetry::ImplementedHybridGPU() {
		return true;
	}

	ArrayXcX ManualSymmetry::getAmplitudesAtPoints(const std::vector<FACC> & relevantQs, FACC theta, FACC phi)
	{
		// First, take orientation of object into account, i.e. change theta and phi to newTheta and newPhi
		FACC st = sin(theta);
		FACC ct = cos(theta);
		FACC sp = sin(phi);
		FACC cp = cos(phi);
		const auto Q = relevantQs[0];

		Eigen::Vector3d Qcart(Q * st*cp, Q * st * sp, Q*ct), Qt;
		Eigen::Matrix3d rot;
		Eigen::Vector3d R(tx, ty, tz);
		Qt = (Qcart.transpose() * RotMat) / Q;

		double newTheta = acos(Qt.z());
		double newPhi = atan2(Qt.y(), Qt.x());

		if (newPhi < 0.0)
			newPhi += M_PI * 2.;

		ArrayXcX phases = (
			std::complex<FACC>(0., 1.) *
			(Qt.dot(R) *
			Eigen::Map<const Eigen::ArrayXd>(relevantQs.data(), relevantQs.size()))
			).exp();

		ArrayXcX reses(relevantQs.size());
		reses.setZero();
		if (GetUseGridWithChildren())
		{
			JacobianSphereGrid* jgrid = dynamic_cast<JacobianSphereGrid*>(grid);

			if (jgrid)
			{
				// Get amplitudes from grid
				return jgrid->getAmplitudesAtPoints(relevantQs, newTheta, newPhi) * phases;
			}
		}

		if (GetUseGridAnyChildren())
		{

			int rows = trans.size();

			ArrayXcX results(relevantQs.size());
			results.setZero();
			Vector3d qnew;

			for (auto& orientation : translationsPerOrientation)
			{
				ArrayXcX res_i(relevantQs.size());
				res_i.setZero();

				qnew = Vector3d(Qt.transpose() * orientation.first).normalized();;

				newTheta = acos(qnew.z());
				newPhi = atan2(qnew.y(), qnew.x());

				if (newPhi < 0.0)
					newPhi += M_PI * 2.;

				for (auto& subAmp : _amps)
				{
					res_i += subAmp->getAmplitudesAtPoints(relevantQs, newTheta, newPhi);
				}
				ArrayXcX phases(relevantQs.size());
				phases.setZero();

				Eigen::Vector3d qDirection(st * cp, st * sp, ct);
				for (auto& trans : orientation.second)
				{
					std::complex<FACC> tPhase = std::complex<FACC>(0., 1.) * (qDirection.dot(trans));
					for (int j = 0; j < relevantQs.size(); j++)
					{
						phases(j) += exp(relevantQs[j] * tPhase);
					}
				}

				results += res_i * phases;
			}

			return results * scale;

		} // if GetUseGridAnyChildren


		return scale * getAmplitudesAtPointsWithoutGrid(newTheta, newPhi, relevantQs, phases);
	}

	std::complex<FACC> ManualSymmetry::getAmplitudeAtPoint(FACC q, FACC theta, FACC phi)
	{
		// First, take orientation of object into account, i.e. change theta and phi to newTheta and newPhi
		FACC st = sin(theta);
		FACC ct = cos(theta);
		FACC sp = sin(phi);
		FACC cp = cos(phi);

		Eigen::Vector3d Qcart(q * st * cp, q * st * sp, q * ct), Qt;
		Eigen::Matrix3d rot;
		Eigen::Vector3d R(tx, ty, tz);
		Qt = (Qcart.transpose() * RotMat) / q;

		double newTheta = acos(Qt.z());
		double newPhi = atan2(Qt.y(), Qt.x());

		if (newPhi < 0.0)
			newPhi += M_PI * 2.;

		std::complex<FACC> phase = exp(std::complex<FACC>(0., 1.) * (Qt.dot(R) * q));

		if (GetUseGridWithChildren())
		{
			JacobianSphereGrid* jgrid = dynamic_cast<JacobianSphereGrid*>(grid);

			if (jgrid)
			{
				// Get amplitudes from grid
				return jgrid->getAmplitudeAtPoint(q, newTheta, newPhi) * phase;
			}
		}

		if (GetUseGridAnyChildren())
		{
			int rows = trans.size();

			Vector3d qnew;
			std::complex<FACC> res;

			for (auto& orientation : translationsPerOrientation)
			{

				qnew = Vector3d(Qt.transpose() * orientation.first).normalized();;

				newTheta = acos(qnew.z());
				newPhi = atan2(qnew.y(), qnew.x());

				if (newPhi < 0.0)
					newPhi += M_PI * 2.;

				for (auto& subAmp : _amps)
				{
					res += subAmp->getAmplitudeAtPoint(q, newTheta, newPhi);
				}

				Eigen::Vector3d qDirection(st * cp, st * sp, ct);
				phase = 0;
				for (auto& trans : orientation.second)
				{
					std::complex<FACC> tPhase = std::complex<FACC>(0., 1.) * (qDirection.dot(trans));
					phase += exp(q * tPhase);
				}

				res *= phase;
			}

			return res * scale;

		} // if GetUseGridAnyChildren


		return scale * getAmplitudeAtPointWithoutGrid(newTheta, newPhi, q, phase);
	}
