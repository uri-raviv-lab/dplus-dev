#ifndef __MEMMANAGER_H
#define __MEMMANAGER_H

#include <ctime>
#include <string>
#include <map>

// So as to not mix up pointers with pointers
typedef void *cpuptr_t;
typedef void *gpuptr_t;

/**
	Algorithm for calculating Amplitude from a tree of AmplitudeModels:
	1) Go to the maximal height (H) on the symmetry tree
	2) For each node of height H-1:
	2.1) Allocate an empty grid (this is the parent node)
	2.2) For each child node:
		2.2.1) Compute node grid (or take from cache if exists; if the node has the same
				grid as the previous node, use grid in memory : else, deallocate(?))
		2.2.2) Add node grid to parent's grid
		2.2.3) Cache node grid[ and deallocate(?)]
	2.3) Cache parent grid
	3) H--
	4) if H = 0 done, else go to 2

	Amplitude Cache Management:
	in the cache directory, the filename will consist of:
		/path/<MD5>-<SHA1>.amp
	where MD5 and SHA1 are the MD5 and SHA1 hashes (respectively) of the following tuple:
		<Container name, Model ID, paramStruct->ToVector(), grid resolution, qMax>
	in the case of PDB files, the hashes will be computed from the processed PDB's contents
	(obtained from PDBReader) and the filename will be:
		/path/pdb/<MD5>-<SHA1>.amp
	in the case of symmetries, the hashes will be computed as follows:
		<Type/ID, parameterArray, list of hashes of nested objects>
	where Type/ID is the name or ID number of the symmetry, parameter array is either the
	locations and rotations of the nested objects in the case of a lack of greater symmetry, or
	the parameters of the symmetry and a reference vector (pivot, location, blah), and the list is
	self explanatory (I hope).

	Files that are currently being written will have an additional lock-file:
		/path/<MD5>-<SHA1>.amp.lock
	If it does not exist, jobs can read from it freely
	In the future, the memory manager should also keep frequently used grids in memory

	Tree-based Parameter Vector:
	* The paramStruct data structure implements a method called ToArray(), which returns
	  a Parameter* array and its length
	* The Job keeps track of all the latest paramStructs (modified by calling SetModelParameters(UID, paramStruct)
	* in the frontend)
	* 
    * A function that accepts <Amplitude*, paramstruct tree> should return the full parameter vector:
	  



	Cache manager eviction:
	Problem

 **/

struct MemBuffer {
protected:
	std::string uuid;      // The name that identifies the buffer (<MD5>-<SHA1>.amp)
	gpuptr_t gpuMem;       // If GPUMem is NULL, the buffer is either cached on RAM or HDD
	cpuptr_t cpuMem;	   // If CPUMem is NULL, the buffer is on the HDD

	size_t size;	
	bool isGPUUpToDate;    // True if latest available data is stored on the GPU 
						   // (no need to update), in case both gpuMem&cpuMem are not null

	unsigned int used;     // For LFU
	time_t       lastUsed; // For LRU

	int refCount;          // Reference counter for automatic garbage collection (if <= 0, free memory)

public:
	MemBuffer() : uuid(""), refCount(1) {}

	static const MemBuffer None; // A non-existent buffer
};


enum EvictionMethod {
	EVICT_LFU = 0, // Least-frequently used buffers
	EVICT_LRU,     // Least-recently used buffers
	EVICT_RR,      // Round-robin (first created, first to evict)
	EVICT_RAND,    // Random eviction (for testing of ineffective caching)
	EVICT_MRU,    // Most-recently used (for testing of ineffective caching)
};

/************************************************************************/
/** Buffer Cache                                                         
	------------
	GPU functions as an L1 cache, CPU as L2 cache and HDD is the main memory
	
**/
/************************************************************************/



// The cache...
class MemoryManager
{
protected:
	size_t memSize, gpumemSize;
	std::string hddPath;
	EvictionMethod eMethod;

	std::map<std::string, MemBuffer> buffers;

	// Default sizes: MEM - 90% of RAM, GPU - 100% of VRAM
	// Default path: ./cache
	// Default eviction method: LFU
	MemoryManager();
	MemoryManager(const MemoryManager& rhs) {}
	void operator=(const MemoryManager& rhs) {}
public:
	static MemoryManager& GetInstance() { 
		static MemoryManager mm;
		return mm;
	}

	// Initialize mem manager with different sizes than default
	void Setup(size_t memory, size_t gpuMemory, const std::string& cachePath, EvictionMethod eviction);

	// Returns MemBuffer::None on failure
	// Increases reference counter on multiple allocations
	// Calls Evict if there's not enough memory on CPU
	MemBuffer Allocate(std::string uuid, size_t sz);

	// Decreases reference counter on free
	void Free(MemBuffer buffer);

	// Returns NULL on failure, increases use count and sets last use to now
	// Calls Evict if there's not enough memory on GPU
	gpuptr_t GetGPUBuffer(std::string uuid);

	// Returns NULL on failure, increases use count and sets last use to now
	// Calls Evict if there's not enough memory on CPU
	cpuptr_t GetCPUBuffer(std::string uuid);

	// Forces garbage collection (???)
	void ForceEvict();
};

#endif
