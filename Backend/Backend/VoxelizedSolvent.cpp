#include "Amplitude.h"
#include <queue>

void PDBAmplitude::CalculateSolventSpace()
{
#define BLUE  0		  // Blue, uninitialized
#define GREEN 1		  // Green, atoms
#define RED   2		  // Red, Probe around the atoms
#define LIGHT_BLUE 3  // Light Blue, Far (bulk) solvent
#define PINK 4		  // Pink, Solvation
#define YELLOW 5	  // Yellow, ummmm....
#define GREY 6	  // Grey, ummmm....
	/**
	* 0) TODO: UPDATE COMMENTS HERE!
	* 1) Allocate space (mark all as 0)
	* 2) Mark all atom voxels as atoms (marked as 1)
	* 3) Mark all neighboring voxels within (rad_atom + 2 * rad_probe) that are not already marked as 1 as (solvation) 2
	* 4) If holes are to be filled, flood fill for outer solvent (3) and mark all holes as atoms; else mark all 0 cells as 3
	* 5) If there is a solvation radius, mark it. i.e. find all voxels that are marked as 3 (bulk solvent) and
	*    neighbor a voxel marked as 4 (solvation) and adjust the thickness of the solvation layer (i.e. if the layer
	*    thickness is greater than the probe, make the layer thicker; if the layer is smaller than the probe, make
	*    layer thinner). After this step, and "stray" voxels that are still marked as 2 (probe) should be marked as 1
	*    (excluded volume). 
	* k) Reduce voxels to irregular boxes
	**/

	// 1) Allocate space (mark all as 0)
	int xS, yS, zS;
	xS = int((xMax - xMin) / voxelStep);
	yS = int((yMax - yMin) / voxelStep);
	zS = int((zMax - zMin) / voxelStep);

	_solvent_space.allocate(xS, yS, zS, voxelStep);

	solventBoxCOM.clear();
	solventBoxDims.clear();
	outerSolventBoxCOM.clear();
	outerSolventBoxDims.clear();

	WriteEigenSlicesToFile("C:\\Delete\\TestSlices\\1_Allocate_");

	// 2) Mark all atom voxels as atoms (marked as 1)
	MarkVoxelsNeighboringAtoms(0.0, 0, GREEN, -1);

	WriteEigenSlicesToFile("C:\\Delete\\TestSlices\\2_Mark_Atoms_");

	// 3) Mark all neighboring voxels within(rad_atom + rad_probe) that are not already marked as 1 as (solvation) 2
	if (solventRad > 0.)
	{
		MarkVoxelsNeighboringAtoms(solventRad + 0. /* GRRRRR, today's whim */ /* 1.0 Angstrom, and that's my final offer. */ /*0.07 Half a &^%$&^ radius of water*/, 0, RED, -1 /*Ignore hydrogen atoms for solvation*/);
		WriteEigenSlicesToFile("C:\\Delete\\TestSlices\\3_SolvationHalf_");

	}
	// 4) If holes are to be filled, flood fill for outer solvent (3) and mark all holes as atoms; else mark all 0 cells as 3
	if (pdb.bFillHoles)
	{
		Floodfill3D(0, 0, 0, 0, LIGHT_BLUE); // From first voxel up to the center of the solvation
		// holes should now be marked as atoms
		_solvent_space.SpaceRef() = (_solvent_space.SpaceRef() == BLUE).select(GREEN, _solvent_space.SpaceRef());
	}
	else
		_solvent_space.SpaceRef() = (_solvent_space.SpaceRef() == 0).select(LIGHT_BLUE, _solvent_space.SpaceRef());

	WriteEigenSlicesToFile("C:\\Delete\\TestSlices\\4_After_Holes_");

	// 5) If there is a solvation radius, mark it
	if (solventRad > 0.)
	{
		MarkLayerOfSolvent(solventRad, LIGHT_BLUE, RED, PINK, RED);
		WriteEigenSlicesToFile("C:\\Delete\\TestSlices\\4Mark1");
		if (solventRad < solvationThickness)
			MarkLayerOfSolvent(fabs(solvationThickness - solventRad), PINK, LIGHT_BLUE, PINK, LIGHT_BLUE);
		else
			MarkLayerOfSolvent(fabs(solvationThickness - solventRad), PINK, LIGHT_BLUE, LIGHT_BLUE, PINK);

		WriteEigenSlicesToFile("C:\\Delete\\TestSlices\\4Mark2");

		// Make sure no atoms were marked as solvation
		MarkVoxelsNeighboringAtoms(0.0, 4, -1, -1);
		// Leftover 2 voxels should be marked as 'excluded volume'
		_solvent_space.SpaceRef() = (_solvent_space.SpaceRef() == 2).select(1, _solvent_space.SpaceRef());
		int pixelsPerNM = 1. / voxelStep;
		for (int i = 0; i < xS; i++)
		{
			auto slice = _solvent_space.SliceX(i);
			for (int i = 0; i < pixelsPerNM; i++)
			{
				slice(1, 1 + i) = 8;
				slice(2, 1 + i) = 8;
				slice(3, 1 + i) = 8;
				slice(4, 1 + i) = 8;
			}
		}
		WriteEigenSlicesToFile("C:\\Delete\\TestSlices\\5_MarkedFullSolvation_");

	}

	//  	for (int i = 0; i < xS; i++)
	//  	{
	//  		std::cout << i << ":\n" << _solvent_space.SliceX(i) << "\n\n";
	//  	}

	auto atomCells = (_solvent_space.SpaceRef() == GREEN).count();
	std::cout << atomCells << " cells out of " <<
		_solvent_space.SpaceRef().size() << " are atoms. " << atomCells * voxelStep * voxelStep * voxelStep << "nm^3\n";

	auto solvationCells = (_solvent_space.SpaceRef() == PINK).count();
	std::cout << solvationCells << " cells out of " <<
		_solvent_space.SpaceRef().size() << " are solvation. " << solvationCells  * voxelStep * voxelStep * voxelStep << "nm^3\n";



	// 4) Reduce voxels to irregular boxes
	ReduceSolventSpaceToIrregularBoxes(solventBoxCOM, solventBoxDims, GREEN, YELLOW);
	int missed = (_solvent_space.SpaceRef() == GREEN).count();
	if (missed != 0)
		std::cout << "Missed " << missed << " excluded volume voxels\n";
	WriteEigenSlicesToFile("C:\\Delete\\TestSlices\\6ReducedExcluded");
	ReduceSolventSpaceToIrregularBoxes(outerSolventBoxCOM, outerSolventBoxDims, PINK, GREY);
	missed = (_solvent_space.SpaceRef() == PINK).count();
	if (missed != 0)
		std::cout << "Missed " << missed << " solvation voxels\n";
	WriteEigenSlicesToFile("C:\\Delete\\TestSlices\\7ReducedSolvation");

	solventBoxCOM_array.resize(solventBoxCOM.size(), Eigen::NoChange);
	solventBoxDims_array.resizeLike(solventBoxCOM_array);
	for (int i = 0; i < solventBoxCOM.size(); i++)
	{
		solventBoxCOM_array.row(i) = Eigen::Array3f(solventBoxCOM[i].x, solventBoxCOM[i].y, solventBoxCOM[i].z);
		solventBoxDims_array.row(i) = float(voxelStep) * Eigen::Array3f(solventBoxDims[i].x, solventBoxDims[i].y, solventBoxDims[i].z);
	}

	int outer_some_number = 0;
	outerSolventBoxCOM_array.resize(outerSolventBoxCOM.size(), Eigen::NoChange);
	outerSolventBoxDims_array.resizeLike(outerSolventBoxCOM_array);
	for (int i = 0; i < outerSolventBoxCOM.size(); i++)
	{
		outerSolventBoxCOM_array.row(i) = Eigen::Array3f(outerSolventBoxCOM[i].x, outerSolventBoxCOM[i].y, outerSolventBoxCOM[i].z);
		outerSolventBoxDims_array.row(i) = float(voxelStep) * Eigen::Array3f(outerSolventBoxDims[i].x, outerSolventBoxDims[i].y, outerSolventBoxDims[i].z);
		outer_some_number += outerSolventBoxDims[i].x * outerSolventBoxDims[i].y * outerSolventBoxDims[i].z;
	}

	std::cout << solventBoxCOM_array.rows() << " boxes as excluded volume.\n";
	std::cout << outerSolventBoxCOM_array.rows() << " boxes as solvation which is comparable to " << outer_some_number << " voxels\n";

	// Sanity Check
	double total_volume = outerSolventBoxDims_array.cast<double>().rowwise().prod().sum();
	std::cout << "Total excluded solvent box volume: " << solventBoxDims_array.cast<double>().rowwise().prod().sum() << "\n";
	std::cout << "Total outer solvent box volume: " << total_volume << "\n";
	std::cout << (_solvent_space.SpaceRef() == YELLOW).count() << " voxels marked as solvent boxes\n";
	std::cout << (_solvent_space.SpaceRef() == GREY).count() << " voxels marked as solvation boxes\n";

// 	int number_of_single_voxels = 0;
// 	std::map<int, int> asafs_histogram, asafs_histogram2;
// 	for (int i = 0; i < outerSolventBoxDims.size(); i++)
// 	{
// 		number_of_single_voxels += int(
// 			outerSolventBoxDims[i].x == 1 &&
// 			outerSolventBoxDims[i].y == 1 &&
// 			outerSolventBoxDims[i].z == 1
// 			);
// 
// 		int num =
// 			outerSolventBoxDims[i].x *
// 			outerSolventBoxDims[i].y *
// 			outerSolventBoxDims[i].z;
// 
// 		asafs_histogram2[outerSolventBoxDims[i].x]++;
// 		asafs_histogram2[outerSolventBoxDims[i].y]++;
// 		asafs_histogram2[outerSolventBoxDims[i].z]++;
// 
// 		asafs_histogram[num]++;
// 	}
// 
// 	std::cout << "\nVolume:\n";
// 
// 	for (const auto& kv : asafs_histogram)
// 	{
// 		std::cout << "[" << kv.first << "]\t" << kv.second << "\n";
// 	}
// 
// 	std::cout << "\nIndices:\n";
// 
// 	for (const auto& kv : asafs_histogram2)
// 	{
// 		std::cout << "[" << kv.first << "]\t" << kv.second << "\n";
// 	}
// 
// 	std::cout << "Number of voxel sized boxes: " << number_of_single_voxels << "\n";

	bSolventLoaded = (solventBoxCOM_array.size() + outerSolventBoxCOM_array.size() > 0);

	_solvent_space.deallocate();
}

void PDBAmplitude::MarkLayerOfSolvent(FACC radius, SolventSpace::ScalarType type, SolventSpace::ScalarType neighbor, SolventSpace::ScalarType totype, SolventSpace::ScalarType fromType)
{
	std::vector< idx > inds;
	int discRad = int(0.5 + (radius) / voxelStep);
	const auto dims = _solvent_space.dimensions();
	inds.reserve(int(dims.matrix().norm()));

	// Find all voxels that are marked `type` and neighbor a voxel marked `neighbor`
	// Ignore the border layer (0 and dims - 1)
	for (uint64_t i = 1; i < dims(0) - 1; i++)
	{
		const auto slice1_1 = _solvent_space.SliceX(i);
		for (uint64_t j = 1; j < dims(1) - 1; j++)
		{
			const auto col = slice1_1.col(j);
			for (uint64_t k = 1; k < dims(2) - 1; k++)
			{
				if (
					col(k) == type && (
					(_solvent_space.SurroundingVoxels(i - 1, j, k) == neighbor).any() ||
					(_solvent_space.SurroundingVoxels(i, j, k) == neighbor).any() ||
					(_solvent_space.SurroundingVoxels(i + 1, j, k) == neighbor).any())
					)
					inds.push_back({ i, j, k });
			} // for k
		} // for j
	} // for i

	// Mark neighbors within radius as `toType`
#pragma omp parallel for
	for (long i = 0; i < inds.size(); i++) {
		int hh = (inds[i])[0], kk = (inds[i])[1], ll = (inds[i])[2];
		const int mEnd = std::min(static_cast<int>(dims[0] - 1), (hh + discRad));
		const int nEnd = std::min(static_cast<int>(dims[1] - 1), (kk + discRad));
		const int pEnd = std::min(static_cast<int>(dims[2] - 1), (ll + discRad));

		for (auto m = std::max(0, hh - discRad); m <= mEnd; m++) {
			auto slice1 = _solvent_space.SliceX(m);

			/*
			const int pSize = 1 + pEnd - p;
			const int nSize = 1 + nEnd - n;
			Eigen::ArrayXXf tmpsqrt = (
			((Eigen::ArrayXi::LinSpaced(pSize, p, pEnd) - ll).square() + sq(m - hh)).eval().transpose().replicate(nSize, 1)
			+ (Eigen::ArrayXi::LinSpaced(nSize, n, nEnd) - kk).square().eval().replicate(1, pSize)
			).eval() .sqrt().cast<float>();
			slice1.block(p, n, pSize, nSize) = (tmpsqrt <= discRad).select(totype, slice1.block(p, n, pSize, nSize));
			*/
			for (auto n = std::max(0, kk - discRad); n <= nEnd; n++)
			{
				auto col = slice1.col(n);
				for (auto p = std::max(0, ll - discRad); p <= pEnd; p++)
				{
					if (col(p) != fromType)
						continue;
					const auto v = sqrt(sq(m - hh) + sq(n - kk) + sq(p - ll));
					if (v <= discRad)
						col(p) = totype;
				} // for p
			} // for n
		} // for m
	} // for


}


void PDBAmplitude::ReduceSolventSpaceToIrregularBoxes(std::vector<fIdx>& boxCOM, std::vector<idx>& boxDims, int designation, int mark_used_as)
{
	size_t current_zero_based_x_length, current_zero_based_y_length, current_zero_based_z_length;
	bool xDone = false, yDone = false, zDone = false;
	auto dims = _solvent_space.dimensions();
	for (size_t x_base = 0; x_base < dims(0); x_base++)
	{
		for (size_t y_base = 0; y_base < dims[1]; y_base++)
		{
			for (size_t z_base = 0; z_base < dims[2]; z_base++)
			{
				if (_solvent_space(x_base, y_base, z_base) == designation)
				{

					current_zero_based_x_length = current_zero_based_y_length = current_zero_based_z_length = 0;
					xDone = yDone = zDone = false;
					while (!(xDone && yDone && zDone))
					{
						if (!xDone)
						{
							current_zero_based_x_length++;
							bool contX = (x_base + current_zero_based_x_length < dims[0]);
							for (size_t plus_y = y_base; contX && plus_y <= y_base + current_zero_based_y_length; plus_y++)
							{
								for (size_t plus_z = z_base; contX && plus_z <= z_base + current_zero_based_z_length; plus_z++)
								{
									contX = (_solvent_space(x_base + current_zero_based_x_length, plus_y, plus_z) == designation);
								} // for plus_z
							} // for plus_y
							if (!contX)
							{
								xDone = true;
								current_zero_based_x_length--;
							}
						} // if xDone
						if (!yDone)
						{
							current_zero_based_y_length++;
							bool contY = (y_base + current_zero_based_y_length < dims[1]);
							for (size_t plus_x = x_base; contY && plus_x <= x_base + current_zero_based_x_length; plus_x++)
							{
								for (size_t plus_z = z_base; contY && plus_z <= z_base + current_zero_based_z_length; plus_z++)
								{
									contY = (_solvent_space(plus_x, y_base + current_zero_based_y_length, plus_z) == designation);
								} // for plus_z
							} // for plus_x

							if (!contY)
							{
								yDone = true;
								current_zero_based_y_length--;
							}
						} // if yDone
						if (!zDone) {
							current_zero_based_z_length++;
							bool contZ = (z_base + current_zero_based_z_length < dims[2]);
							for (size_t plus_x = x_base; contZ && plus_x <= x_base + current_zero_based_x_length; plus_x++)
							{
								for (size_t plus_y = y_base; contZ && plus_y <= y_base + current_zero_based_y_length; plus_y++)
								{
									contZ = (_solvent_space(plus_x, plus_y, z_base + current_zero_based_z_length) == designation);
								} // for plus_y
							} // for plus_x

							if (!contZ) {
								zDone = true;
								current_zero_based_z_length--;
							}
						} // if zDone

					} // while !(xDone && yDone && zDone)
					// Add coordinates to class variable
					// Add dimensions to class variable
					// Mark all positions as <mark_used_as>
					fIdx coord = {
						float(xMin + (FACC(x_base) + (FACC(current_zero_based_x_length /*?+ 1*/) / 2.0)) * voxelStep),
						float(yMin + (FACC(y_base) + (FACC(current_zero_based_y_length /*?+ 1*/) / 2.0)) * voxelStep),
						float(zMin + (FACC(z_base) + (FACC(current_zero_based_z_length /*?+ 1*/) / 2.0)) * voxelStep)
					};
					boxCOM.push_back(coord);

					idx dims = { current_zero_based_x_length + 1, current_zero_based_y_length + 1, current_zero_based_z_length + 1 };
					boxDims.push_back(dims);

					for (size_t hh = x_base; hh <= x_base + current_zero_based_x_length; hh++) {
						auto slice1_1 = _solvent_space.SliceX(hh);
						for (size_t kk = y_base; kk <= y_base + current_zero_based_y_length; kk++) {
							auto slice2_1 = slice1_1.col(kk);
							for (size_t ll = z_base; ll <= z_base + current_zero_based_z_length; ll++) {
								slice2_1(ll) = mark_used_as;
							} // for ll
						} // for kk
					} // for hh


				} // if col(l) == designation
			} // z_base
		} // y_base
	} // z_base

}

void PDBAmplitude::MarkVoxelsNeighboringAtoms(FACC rSol, SolventSpace::ScalarType from, SolventSpace::ScalarType to, int ignoreIndex)
{
	int xSz = (int)pdb.x.size();
	for (int i = 0; i < xSz; i++) {
		if (pdb.atmInd[i] == ignoreIndex)
			continue;
		const double ddist = ((*pdb.rad)[pdb.atmInd[i]]) + rSol;
		const auto dm = _solvent_space.dimensions();
		//std::cout << ddist << "\n";
		int dist = 2 + int(ddist / voxelStep);
		int xC = int((pdb.x[i] - xMin) / voxelStep);
		int yC = int((pdb.y[i] - yMin) / voxelStep);
		int zC = int((pdb.z[i] - zMin) / voxelStep);
		for (int h = std::max(0, xC - dist); h < std::min(xC + dist, int(dm(0))); h++) {
			auto sliceX = _solvent_space.SliceX(h);
			for (int k = std::max(0, yC - dist); k < std::min(yC + dist, int(dm(1))); k++) {
				auto col = sliceX.col(k);
				for (int l = std::max(0, zC - dist); l < std::min(zC + dist, int(dm(2))); l++) {
					if (col(l) == from && // Check to make sure that the voxel has not yet been marked
						sqrt(
						sq(xMin + (FACC(h) * voxelStep) - pdb.x[i]) +
						sq(yMin + (FACC(k) * voxelStep) - pdb.y[i]) +
						sq(zMin + (FACC(l) * voxelStep) - pdb.z[i]))
						<= ddist
						)
					{
						col(l) = to;
					}
				} //for l
			} // for k
		} // for h
	} // for i
}

void PDBAmplitude::Floodfill3D(int i, int j, int k, int from, int to)
{
	std::queue<sIdx> qu;

	if (_solvent_space(i, j, k) != from) {
		return;
	}

	sIdx pb = { i, j, k };
	qu.push(pb);
	int maxSize = 0, siz = int(qu.size());
	SolventSpace::Dimensions shape = _solvent_space.dimensions();
	_solvent_space(i, j, k) = to;

	while (!qu.empty())
	{
		pb = qu.front();
		qu.pop();
		siz = int(qu.size());
		i = pb[0]; j = pb[1]; k = pb[2];
		const int iiEnd = std::min(int(shape[0] - 1), i + 1);
		for (int ii = std::max(0, i - 1); ii <= iiEnd; ii++)
		{
			auto slice1 = _solvent_space.SliceX(ii);
			const int jjEnd = std::min(int(shape[1] - 1), j + 1);
			for (int jj = std::max(0, j - 1); jj <= jjEnd; jj++)
			{
				auto col = slice1.col(jj);
				const int kkEnd = std::min(int(shape[2] - 1), k + 1);
				for (int kk = std::max(0, k - 1); kk <= kkEnd; kk++)
				{
					// if is marked
					if (col(kk) != from)
					{
						continue;
					}
					qu.push({ ii, jj, kk });
					col(kk) = to;
				}	// for(int kk = k - 1; kk <= k + 1; kk++)
			}	// for(int jj = j - 1; jj <= j + 1; jj++)
		}	// for(int ii = i - 1; ii <= i + 1; ii++)
	}	// while(!qu.empty())
}

SolventSpace::ScalarType& SolventSpace::operator()(size_t x, size_t y, size_t z)
{
	assert(x >= 0);
	assert(y >= 0);
	assert(z >= 0);

	assert(x < _x_size);
	assert(y < _y_size);
	assert(z < _z_size);

	return _solvent_space(x * _zy_plane + y * _z_size + z);
}

void SolventSpace::allocate(size_t x, size_t y, size_t z, float voxel_length)
{
	assert(x > 0);
	assert(y > 0);
	assert(z > 0);

	// Align to 16 bytes
	const int numelements = 16 / sizeof(ScalarType);

	_z_size = z + (numelements - 1 - (z + numelements - 1) % numelements);
	_y_size = y;
	_x_size = x;

	_zy_plane = _z_size * _y_size;

	_solvent_space = array_t::Zero(_x_size, _z_size * _y_size);
	_voxel_length = voxel_length;
}

Eigen::Map<SolventSpace::array_t, Eigen::AlignmentType::Aligned> SolventSpace::SliceX(size_t x)
{
	return Eigen::Map<array_t, Eigen::AlignmentType::Aligned>(_solvent_space.data() + x * _zy_plane, _z_size, _y_size);
}

Eigen::Map<SolventSpace::array_t, 0, Eigen::Stride<Eigen::Dynamic, 1>> SolventSpace::SurroundingVoxels(size_t x, size_t y, size_t z)
{
	Eigen::Map<array_t, 0, Eigen::Stride<Eigen::Dynamic, 1>> box9(
		_solvent_space.data() + x * _zy_plane + (y-1) * _z_size + (z-1),
		3, 3, Eigen::Stride<Eigen::Dynamic, 1>(_z_size, 1));

	return box9;
}

SolventSpace::array_t& SolventSpace::SpaceRef()
{
	return _solvent_space;
}

SolventSpace::Dimensions SolventSpace::dimensions()
{
	return Dimensions(_x_size, _y_size, _z_size);
}
void SolventSpace::deallocate()
{
	_solvent_space.resize(0, 0);
}