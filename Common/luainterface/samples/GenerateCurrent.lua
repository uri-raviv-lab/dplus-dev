function recprint(tree, prefix)
	for k,v in pairs(tree) do 		
		if type(v) == "table" then
			recprint(v, "\t"..prefix..k..".");
		else
			print(prefix..k..": "..tostring(v));
		end
	end
end


odata = dplus.generate();

-- Opens the console
dplus.openconsole();

ptree = dplus.getparametertree();

recprint(ptree, "");

ptree.Models[1].Parameters[1][1] = 8;
ptree.Models[1].Parameters[2][1] = 4.01;


msgbox("A");

dplus.closeconsole();

ndata = dplus.generate(ptree);

dplus.showgraph(odata);
dplus.showgraph(ndata, -1, "blue");

