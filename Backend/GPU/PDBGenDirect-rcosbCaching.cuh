#pragma once

#include "GPUInterface.h"

class GPUDirectCalculator : public IGPUCalculator
{
protected:      
	// Inputs
    float2 *d_angles;
    float  *d_qPoints;
    size_t _numAngles, _numQ;
    size_t _anglePitch;
	
	float4 *d_atomLocs;
	float *d_affs;
	unsigned int *h_atomsPerIon;
	size_t _numAtoms,  _maxNumAtoms;

	size_t _numCoeffs, _maxNumCoeffs;

	// Processed data and outputs
	float *d_rcosb;
	size_t _rcosbPitch;
	double2 *d_amp;
	double *d_transamp, *d_intensity;
	int *d_intensityIndices; // Necessary for thrust
	size_t _ampPitch;
    
public:
	GPUDirectCalculator();
    virtual ~GPUDirectCalculator();
    
    // Should be called once
    virtual bool Initialize(const float2 *angles, size_t numAngles,
                            const float *qPoints, size_t numQ,
                            size_t maxNumAtoms, size_t maxNumCoeffs);

    // Sets the current PDB
    virtual bool SetPDB(const float4 *atomLocs, const unsigned char *ionInd,
                        size_t numAtoms, const float *coeffs, const unsigned int *atomsPerIon,
                        size_t numCoeffs);
	
	virtual bool ClearAmplitude();

    // Adds to "data" the set PDB, with the given translation
    // and rotation
    virtual bool AppendAmplitude(float3 translation, float3 rotation);

    
    virtual bool ComputeIntensity(double *data);
};

