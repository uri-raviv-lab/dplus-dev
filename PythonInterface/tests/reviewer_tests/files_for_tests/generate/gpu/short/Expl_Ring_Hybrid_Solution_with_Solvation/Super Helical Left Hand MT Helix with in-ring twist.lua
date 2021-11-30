Information = {
	Name = "Super Helical Left Hand MT Helix with a twist - order of rotation changed",
	Type = "Symmetry",
	NLP = 1,
	MinLayers = 9,
	MaxLayers = 9,
};

function Populate(p, nlayers)	 
	if (p == nil or nlayers ~= 9 or table.getn(p[1]) ~= 1) then				
		error("Parameter matrix must be 7x1, it is " .. nlayers .. "x" .. table.getn(p[1]));
	end
	
	--error("Params: " .. p[1][1] .. ", " .. p[2][1] .. ", " .. p[3][1]);

	radius				= p[1][1];
	pitch				= p[2][1];
	unitsPerPitch		= p[3][1];
	unitsInPitch		= p[4][1];
	discreteHeight		= p[5][1];
	numHelixStarts		= p[6][1];
	superHelicalPitch	= p[7][1];
	RingTwistAlpha		= p[8][1];
	RingTwistBeta		= p[9][1];
	
	longitudinalSpacing = (pitch * 2.0 / numHelixStarts);
	
	angle		= 2.0 * math.pi / unitsPerPitch;
	if(superHelicalPitch >  0.00001) then
		-- angleShift is the amount in radians per dimer by which single longitudinal layer misses the 2\pi mark
		angleShift = (2.0 * math.pi * longitudinalSpacing) / (superHelicalPitch * unitsPerPitch);
	else
		angleShift = 0.0;
	end
	
	res = {};
	n = 1;
	m = 1;
	
	ind = 0;

	for heightInd = 0, discreteHeight-1 do

		initialLayerShift = math.fmod(heightInd * (2.0 / numHelixStarts) * angleShift * unitsPerPitch, 2. * math.pi);
		
		hUnitsPerPitch = 2.0 * math.pi / (angle + angleShift * unitsPerPitch);

		initialZShift = heightInd * longitudinalSpacing;
		
		for inPitchInd = 0, unitsInPitch - 1 do
			theta = initialLayerShift + inPitchInd * (angle + angleShift);
			x		= radius * math.sin(theta);
			y		= radius * math.cos(theta);
			z		= initialZShift + (inPitchInd / unitsPerPitch) * pitch;

			-- the order of rotation will be first beta, then gamma and last alpha. to change it use the relevant rotation matrix values in the theta calculation

			alphaBetaFirst	= (inPitchInd * RingTwistAlpha) / 180 * math.pi;
			betaBetaFirst = inPitchInd * RingTwistBeta / 180 * math.pi;
			gammaBetaFirst	= (90. - 180. * theta / math.pi) / 180 * math.pi;

			epsilon = math.pow(10.,-5.);
			
--  		if (1.0 - np.abs(Rmatrix[0,2])) > epsilon:

			if ((1.0 - math.abs(math.cos(gammaBetaFirst) * math.sin(betaBetaFirst))) > epsilon) then
				
--  			# finding beta: 			

--		        thet1 = np.arcsin(Rmatrix[0,2])

				beta = math.asin(math.cos(gammaBetaFirst) * math.sin(betaBetaFirst));

--				# finding alpha:

--		        psi1 = np.arctan2(-Rmatrix[1,2]/np.cos(thet1),Rmatrix[2,2]/np.cos(thet1))

				alpha = math.atan2(-(-math.cos(betaBetaFirst) * math.sin(alphaBetaFirst) + math.cos(alphaBetaFirst) * math.sin(betaBetaFirst) * math.sin(gammaBetaFirst)) / math.cos(beta), 
				(math.cos(alphaBetaFirst) * math.cos(betaBetaFirst) + math.sin(alphaBetaFirst) * math.sin(betaBetaFirst) * math.sin(gammaBetaFirst)) / math.cos(beta));
			

--				# finding gamma: 

--				phi1 = np.arctan2(-Rmatrix[0,1]/np.cos(thet1),Rmatrix[0,0]/np.cos(thet1))

				gamma = math.atan2(-(-math.sin(gammaBetaFirst)) / math.cos(beta),(math.cos(betaBetaFirst) * math.cos(gammaBetaFirst)) / math.cos(beta));

			else

				gamma = 0.0;

				if ((1.0 - math.cos(gammaBetaFirst) * math.sin(betaBetaFirst)) < epsilon) then

					beta = math.pi / 2.;

					alpha = gamma + math.atan2(math.sin(alphaBetaFirst) * math.sin(betaBetaFirst) + math.cos(alphaBetaFirst) * math.cos(betaBetaFirst) * math.sin(gammaBetaFirst), 
					-(-math.cos(alphaBetaFirst) * math.sin(betaBetaFirst) + math.cos(betaBetaFirst) * math.sin(alphaBetaFisrt) * math.sin(gammaBetaFirst)));

				else

					beta =  -math.pi / 2.;

					alpha = - gamma + math.atan2(- (math.sin(alphaBetaFirst) * math.sin(betaBetaFirst) + math.cos(alphaBetaFirst) * math.cos(betaBetaFirst) * math.sin(gammaBetaFirst)), 
					(-math.cos(alphaBetaFirst) * math.sin(betaBetaFirst) + math.cos(betaBetaFirst) * math.sin(alphaBetaFisrt) * math.sin(gammaBetaFirst)));
					
				end

			end

			--print(heightInd .. " " .. inPitchInd .. ": [" .. x .. ", " .. y ..  ", " .. z .. "]");
			alpha = alpha * 180 / math.pi;
			beta = beta * 180 / math.pi;
			gamma = gamma * 180 / math.pi;
			res[ind+1] = {x,y,z,alpha,beta,gamma};
			ind = ind + 1;
		end
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
		return "Discrete Height";
	elseif index == 5 then
		return "# Helix Starts";
	elseif index == 6 then
		return "Super Helical Pitch";
	elseif index == 7 then
		return "Ring twist alpha";
	elseif index == 8 then
		return "Ring twist beta";
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
		return 18;
	elseif layer == 1 then
		return 0;
	elseif layer == 2 then
		return 14.0;
	elseif layer == 3 then
		return 14.0;
	elseif layer == 4 then
		return 1;
	elseif layer == 5 then
		return 1;
	elseif layer == 6 then
		return 0.0;
	elseif layer == 7 then
		return 0.0;
	elseif layer == 8 then
		return 0.0;
	end
end

	