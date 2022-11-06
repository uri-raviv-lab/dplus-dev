#pragma once

#include "Common.h"
#include "ModelUI.h"

using Aga::Controls::Tree::Node;
using LuaInterface::Lua;
using System::Runtime::InteropServices::GCHandle;

namespace DPlus {

	public enum class EntityType {
		TYPE_PDB,
		TYPE_EPDB,
		TYPE_AMPGRID,
		TYPE_PRIMITIVE,
		TYPE_SYMMETRY
	};

	public value struct LocationRotationCLI {
		double x, y, z;
		/// RADIAN!!!
		double radAlpha, radBeta, radGamma;

		LocationRotationCLI(LocationRotation other) {
			x = other.x; y = other.y; z = other.z; 
			radAlpha = other.alpha; radBeta = other.beta; radGamma = other.gamma; 
		}
	};

	public ref class Entity : Node {
	protected:
		paramStruct *params;
		ModelUI *model;		
		
	public:
		double x()     { if(params) return params->x.value;     else return 0.0; }
		double y()     { if(params) return params->y.value;     else return 0.0; }
		double z()     { if(params) return params->z.value;     else return 0.0; }
		Radian alpha() { if(params) return Radian(params->alpha.value); else return Radian(); }
		Radian beta()  { if(params) return Radian(params->beta.value ); else return Radian(); }
		Radian gamma() { if(params) return Radian(params->gamma.value); else return Radian(); }

		void SetX(double x)         { if(params) params->x.value = x;          }
		void SetY(double y)         { if(params) params->y.value = y;          }
		void SetZ(double z)         { if(params) params->z.value = z;          }
		void SetAlpha(Radian alpha) { if(params) params->alpha.value = alpha;  }
		void SetBeta (Radian beta)  { if(params) params->beta.value  = beta;   }
		void SetGamma(Radian gamma) { if(params) params->gamma.value = gamma;  }

		void SetUseGrid(bool bUseG);
		bool IsParentUseGrid();
		
		void SetXMut(bool val)         { if(params) params->x.isMutable = val;  }
		void SetYMut(bool val)         { if(params) params->y.isMutable = val;  }
		void SetZMut(bool val)         { if(params) params->z.isMutable = val;  }
		void SetAlphaMut(bool val) { if(params) params->alpha.isMutable = val;  }
		void SetBetaMut (bool val) { if(params) params->beta.isMutable  = val;  }
		void SetGammaMut(bool val) { if(params) params->gamma.isMutable = val;  }

		EntityType type;
		System::String ^ modelName;
		System::String ^ displayName;

		// For PDBs and amplitude grids
		System::String ^filename, ^modelfile, ^anomfilename;
		std::vector<float>* xs;
		std::vector<float>* ys;
		std::vector<float>* zs;
		std::vector<u8>* atoms;
		bool bCentered;

		FrontendComm *frontend;
		JobPtr job;
		ModelPtr BackendModel;

		// For primitives and amplitudes
		LevelOfDetail currentLOD;
		renderFunc render;		
		unsigned int renderDList, colorCodedDList;

		// For symmetries
		symmetryRenderFunc symmrender;
		System::Collections::Generic::List<LocationRotationCLI> ^locs;

		// For scripted models and symmetries (keeps the functions from being
		// garbage-collected)
		Lua ^modelContext;
		GCHandle hlnf, hlpnf, hdpvf, hipaf;

		bool selected;

		virtual property System::String ^Text {
			System::String ^get() override {
				if (displayName == "")
					return modelName;
				return modelName + " ("+displayName+")";
			}
		}

		virtual System::String ^ToString() override {
			if (displayName == "")
				return modelName;
			return modelName + " (" + displayName + ")";
		}

		static int i = 0;
		property ModelUI *modelUI {
			ModelUI *get() { return model; }
			void set(ModelUI *value) { 
				model = value; 
				if(model) // If model exists, modify the entity's display name
					modelName = gcnew System::String(model->GetName().c_str());
				displayName = "";
				i++;
			}
		}

		Entity();
		~Entity();

		void SetParameters(const paramStruct& parConst, LevelOfDetail lod);

		void Invalidate(LevelOfDetail lod, bool bInvalidateParents);
		
		void validateConstrains(System::Collections::Generic::List<Entity^>^ invalidVec);

		System::String^ InvalidParamsString();

		void FixConstrains();

		void Invalidate(LevelOfDetail lod) { Invalidate(lod, true); }

		paramStruct GetParameters() {
			return *params;
		}
	};

}

