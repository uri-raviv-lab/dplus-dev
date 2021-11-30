

myTree = dplus.getparametertree ()
dummyVar = PrintKeys(myTree)
--[[Prints :
	Size : 4
	Scale
	ScaleMut
	Geometry
	Populations
]]
dummyVar = PrintKeys(myTree.Populations)
--[[ Prints :
	Size : 1
	1
]]
dummyVar = PrintKeys(myTree.Populations[ 1] )
--[[ Prints :
	Size : 3
	PopulationSizeMut
	PopulationSize
	Models
]]
--etc .


res = dplus.generate ()
ddd = dplus.writedata(saveFileName, res)
if not ddd then
	print(”There was a problem with fi l e ” . . saveFileName . . ”. \n”) ;
end

	