Information = {
	Name = "Square lattice in XY", -- This is the name that will be displayed in the Domain View
	Type = "Symmetry", --This is the type , should be "Symmetry" for scripted symmetries
	NLP = 2, --Number of Layer Parameters: The number of parameters per layer
	MinLayers = 2, --The minimal number of layers (<= MaxLayers)
	MaxLayers = 2, --The maximal number of layers (>= MinLayers)
};

function Populate(p, nlayers)	 
-- This is really just a sanity check , but doesn't hurt to add it.
	if (p == nil or nlayers ~= 2 or table.getn(p[1]) ~= 2) then				
		error("Parameter matrix must be 2x2, it is " .. nlayers .. "x" .. table.getn(p[1]));
	end
	
	--Create meaningful names

		DistanceX			 = p[1][1];
		DistanceY			 = p[2][1];
		NumberOfRepetitionsX = p[1][2];
		NumberOfRepetitionsY = p[2][2];
	
	
	res = {};
	
	for n = 1, NumberOfRepetitionsY	 do
		for m = 1, NumberOfRepetitionsX	 do
			res[m+(n-1)*NumberOfRepetitionsX] = {DistanceX*(m-1)-((NumberOfRepetitionsX/2-0.5)*DistanceX),DistanceY*(n-1)-((NumberOfRepetitionsY/2-0.5)*DistanceY),0,0,0,0};
		end
	end
	return res;

end
	
	

-----------------------------------------------------
-- UI

-- Optional display parameters
function GetLayerName(index)
	if index == 0 then
		return "X";
	elseif index == 1 then
		return "Y";
	
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
			return 5;
		end
	elseif layer == 1 then
		if layerParam == 0 then
			return 10;
		elseif layerParam == 1 then
			return 5;
		end
	
	end
end
	
	


	