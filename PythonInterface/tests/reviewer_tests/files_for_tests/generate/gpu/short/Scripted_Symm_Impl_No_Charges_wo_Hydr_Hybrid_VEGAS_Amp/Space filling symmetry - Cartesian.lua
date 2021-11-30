Information = {
	Name = "Space Filling Symmetry - Cartesian",
	Type = "Symmetry",
	NLP = 1,
	MinLayers = 6,
	MaxLayers = 6,
};

function Populate(p, nlayers)	 
	if (p == nil or nlayers ~= 6 or table.getn(p[1]) ~= 1) then				
		error("Parameter matrix must be 6x1, it is " .. nlayers .. "x" .. table.getn(p[1]));
	end
	
	xRepeat				= p[1][1];
	xDistance			= p[2][1];
	yRepeat				= p[3][1];
	yDistance			= p[4][1];
	zRepeat				= p[5][1];
	zDistance			= p[6][1];
	
	res = {};
	
	ind = 0;

	for xInd = 0, xRepeat-1 do

		for yInd = 0, yRepeat-1 do
		
			for zInd = 0, zRepeat-1 do
			
				x = xInd * xDistance;
				y = yInd * yDistance;
				z = zInd * zDistance;
				
				alpha = 0;
				beta = 0;
				gamma = 0;
				res[ind+1] = {x,y,z,alpha,beta,gamma};
				ind = ind + 1;
				
			end
			
		end
		
	end
	
	return res;

end

-----------------------------------------------------
-- UI

-- Optional display parameters
function GetLayerName(index)
	if index == 0 then
		return "X - number of repeats";
	elseif index == 1 then
		return "X - space apart";
	elseif index == 2 then
		return "Y - number of repeats";
	elseif index == 3 then
		return "Y - space apart"
	elseif index == 4 then
		return "Z - number of repeats";
	elseif index == 5 then
		return "Z - space apart"
	else	
		return "N/A";
	end
end
function GetLayerParameterName(index)
	if index == 0 then
		return "Parameter";
	else
		return "N/A"
	end
end
	
function IsParamApplicable(layer, layerParam)
	return true;
end

function GetDefaultValue(layer, layerParam)
	if layer == 0 then
		return 3;
	elseif layer == 1 then
		return 4;
	elseif layer == 2 then
		return 2;
	elseif layer == 3 then
		return 4;
	elseif layer == 4 then
		return 3;
	elseif layer == 5 then
		return 8.13;
	end
end

	