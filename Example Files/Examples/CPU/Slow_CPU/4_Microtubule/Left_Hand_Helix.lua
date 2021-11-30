Information = {
	Name = "Left Hand Helix",
	Type = "Symmetry",
	NLP = 1,
	MinLayers = 7,
	MaxLayers = 7,
};

function Populate(p, nlayers)	 
	if (p == nil or nlayers ~= 7 or table.getn(p[1]) ~= 1) then				
		error("Parameter matrix must be 7x1, it is " .. nlayers .. "x" .. table.getn(p[1]));
	end
	
	radius				= p[1][1];
	pitch				= p[2][1];
	unitsPerPitch		= p[3][1];
	unitsInPitch		= p[4][1];
	startAt				= p[5][1];
	discreteHeight		= p[6][1];
	numHelixStarts		= p[7][1];
	
	longitudinalSpacing = (pitch * 2.0 / numHelixStarts);
	
	angle		= 2.0 * math.pi / unitsPerPitch;
	
	res = {};
	n = 1;
	m = 1;
	
	ind = 0;

	for heightInd = 0, discreteHeight-1 do

		hUnitsPerPitch = 2.0 * math.pi / (angle);

		initialZShift = heightInd * longitudinalSpacing;
		
		for inPitchInd = startAt, unitsInPitch - 1 do --Note - this loop skips the first "startAt" points in each pitch!
			theta 	= inPitchInd * angle;
			x		= radius * math.sin(theta);
			y		= radius * math.cos(theta);
			z		= initialZShift + (inPitchInd / unitsPerPitch) * pitch;

			alpha = 0;
			beta = 0;
			gamma	= 90. - 180. * theta / math.pi;
			res[ind+1] = {x,y,z,alpha,beta,gamma};
			ind = ind + 1;
		end
	end
	
	resLen = table.getn(res);
	
	xMean = 0;
	yMean = 0;
	zMean = 0;
	
	for k = 1,resLen do
		xMean = xMean + res[k][1] / resLen;
		yMean = yMean + res[k][2] / resLen;
		zMean = zMean + res[k][3] / resLen;
	end
	
	for k = 1,resLen do
		res[k][1] = res[k][1] - xMean;
		res[k][2] = res[k][2] - yMean;
		res[k][3] = res[k][3] - zMean;
	end
	
	return res;

end

-----------------------------------------------------
-- UI

-- Optional display parameters
function GetLayerName(index)
	if index == 0 then
		return "Radius";
	elseif index == 1 then
		return "Pitch";
	elseif index == 2 then
		return "Units per Pitch";
	elseif index == 3 then
		return "Units in Pitch";
	elseif index == 4 then
		return "Units to skip in pitch";
	elseif index == 5 then
		return "Discrete Height";
	elseif index == 6 then
		return "# Helix Starts";
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
		return 11.94;
	elseif layer == 1 then
		return 12.195;
	elseif layer == 2 then
		return 14.0;
	elseif layer == 3 then
		return 14.0;
	elseif layer == 4 then
		return 0.0;
	elseif layer == 5 then
		return 1;
	elseif layer == 6 then
		return 3;
	end
end

	