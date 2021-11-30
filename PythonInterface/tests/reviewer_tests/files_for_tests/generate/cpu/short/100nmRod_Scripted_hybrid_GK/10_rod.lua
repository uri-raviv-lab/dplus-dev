Information = {
	Name = "lattice in z", -- This is the name that will be displayed in the Domain View
	Type = "Symmetry", --This is the type , should be "Symmetry" for scripted symmetries
	NLP = 2, --Number of Layer Parameters: The number of parameters per layer
	MinLayers = 1, --The minimal number of layers (<= MaxLayers)
	MaxLayers = 1, --The maximal number of layers (>= MinLayers)
};

function Populate(p, nlayers)	 
-- This is really just a sanity check , but doesn't hurt to add it.
	if (p == nil or nlayers ~= 1 or table.getn(p[1]) ~= 2) then				
		error("Parameter matrix must be 2x2, it is " .. nlayers .. "x" .. table.getn(p[1]));
	end
	
	--Create meaningful names

		DistanceZ			 = p[1][1];
		NumberOfRepetitionsZ = p[1][2];
	
	
	
	res = {};
	
	for n = 1, NumberOfRepetitionsZ	 do
				res[n] = {0,0,DistanceZ*(n-1)-((NumberOfRepetitionsZ/2-0.5)*DistanceZ),0,0,0};
		
	end
	return res;

end
	
	

-----------------------------------------------------
-- UI

-- Optional display parameters
function GetLayerName(index)
	if index == 0 then
		return "Z";
		
	else	
		return "N/A";
	end
end
function GetLayerParameterName(index)
	if index == 0 then
		return "distance";
		elseif index == 1 then
		return "Repetitions";
	else
		return "N/A"
	end
end
	
function IsParamApplicable(layer, layerParam)
	return true;
end


function GetDefaultValue(layer, layerParam)
	if layer == 0 then
		if layerParam == 0 then
			return 10;
		elseif layerParam == 1 then
			return 10;
		end	
	end
end
	
	


	