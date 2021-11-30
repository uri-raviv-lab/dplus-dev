#define _USE_MATH_DEFINES
#include "Eigen/Core"

#include "edprofile.h"

#include "graphtoolkit.h"
#include <vector>

using namespace Eigen;
using std::vector;

void MakeSymmetricProfile(graphLine *graphs, int ind) {
	if(graphs[ind].x.size() == 0)
		return;

	// Finding the first positive spot and erasing all negative values from the
	// graph
	for(unsigned int j = 0; j < graphs[ind].x.size() - 1; j++) {
		if(graphs[ind].x.at(j + 1) >= 0.0 && graphs[ind].x.at(j) < 0.0) {
			graphs[ind].x.erase(graphs[ind].x.begin(), graphs[ind].x.begin() + j);
			graphs[ind].y.erase(graphs[ind].y.begin(), graphs[ind].y.begin() + j);
			break;
		}
	}

	// Make the graph symmetric by doubling the number of points and mirroring
	// the profile
	int s = graphs[ind].x.size();
	for(int i = 0; i < s - 1; i++) {
		graphs[ind].x.insert(graphs[ind].x.begin(), -graphs[ind].x.at((2 * i) + 1));
		graphs[ind].y.insert(graphs[ind].y.begin(), graphs[ind].y.at((2 * i) + 1));
	}
}

void generateEDProfile(std::vector< std::vector<Parameter> > p,
					   struct graphLine *graphs, EDProfile profile) {
	graphs[0].color = RGB(200, 123, 0);
	graphs[0].legendKey = "Electron Density";

	if(p.size() < 2)
		return;

	std::vector<Parameter> r = p[0], ed = p[1];

	if(r.size() == 0 || r.size() != ed.size())
		return;

	for(unsigned int i = 0; i < r.size(); i++)
		if(r[i].value < 0.0) r[i].value = 0.0;

	for(unsigned int i = 1; i < ed.size(); i++)
		if(ed[i].value < 0.0) ed[i].value = ed[0].value;

	graphs[0].x.push_back(0.0);
	graphs[0].y.push_back(ed.front().value);

/* TODO::EDP
	EDPFunction *func = profile.func;
	// Create a special function for drawing
	if(!func && profile.shape != DISCRETE) {

		int nlp = p.size(),	nLayers = r.size();

		MatrixXd params = MatrixXd::Zero(nLayers, nlp);

		// Update the ED profile function
		for(int i = 0; i < nlp; i++)
			for(int j = 0; j < nLayers; j++)
				params(j, i) = p[i][j].value;

		func = ProfileFromShape(profile.shape, params);
	}

	// Other ED profiles
	if(func) {
			int layers = r.size();
			std::vector<double> resy, resx;

			// Use the upper limit of the ED profile function to draw the profile
			double upper = func->GetUpperLimit();
	
			const int points = 1000;
			for(int xi = 0; xi < points; xi++) {
				double x = upper * (double)xi / double(points);
				if(x < 0.0)
					continue;
				resx.push_back(x);
				resy.push_back(func->Evaluate(x));
			}

			graphs[0].x = resx;
			graphs[0].y = resy;

			// Highlighting the discrete steps along the way
			double unused;
			MatrixXd steps = func->MakeSteps(0, 0, 0, unused);
			for(int i = 1; i < steps.rows(); i++)
				steps(i, 0) += steps(i - 1, 0);

			// If this is a simulation, draw steps
			if(profile.func) {
				// First point (at 0.0)
				graphs[2].x.push_back(0.0);
				graphs[2].y.push_back(steps(0, 1));

				// Plotting all ED points
				for(int i = 0; i < steps.rows(); i++) {
					graphs[2].x.push_back(steps(i, 0));
					graphs[2].y.push_back(steps(i, 1));
					if(i < steps.rows() - 1) {
						graphs[2].x.push_back(steps(i, 0));
						graphs[2].y.push_back(steps(i + 1, 1));
					} else {
						graphs[2].x.push_back(steps(i, 0));
						graphs[2].y.push_back(steps(i, 1));
						graphs[2].x.push_back(steps(i, 0) * 1.2);
						graphs[2].y.push_back(steps(i, 1));
					}
				}
			}

			// Delete the newly-created ED profile
			if(!profile.func)
				delete func;


	} else if(profile.shape == DISCRETE)
*/
	{
		//Discrete ED
		for(unsigned int i = 1; i < r.size(); i++)
			r.at(i).value += r.at(i - 1).value;

		for(unsigned int i = 0; i < r.size(); i++) {
			graphs[0].x.push_back(r.at(i).value);
			graphs[0].y.push_back(ed.at(i).value);
			if(i < r.size() - 1) {
				graphs[0].x.push_back(r.at(i).value);
				graphs[0].y.push_back(ed.at(i + 1).value);
			} else {
				graphs[0].x.push_back(r.at(i).value);
				graphs[0].y.push_back(ed.front().value);
				graphs[0].x.push_back(r.at(i).value + r.at(i).value * 0.2);
				graphs[0].y.push_back(ed.front().value);
			}
		}
	}

	if(profile.type == SYMMETRIC) {  // Symmetric around x=0.0
		MakeSymmetricProfile(graphs, 0);
		MakeSymmetricProfile(graphs, 2);
	}

	graphs[2].color = RGB(47, 50, 125);
	graphs[2].legendKey = "Discrete Step Simulation";

	graphs[1].color = RGB(37, 20, 225);
	graphs[1].legendKey = "Solvent";

	graphs[1].x.push_back(graphs[0].x.front());
	graphs[1].x.push_back(graphs[0].x.back());
	graphs[1].y.push_back(ed.front().value);
	graphs[1].y.push_back(ed.front().value);
}


std::pair<double, double> calcEDIntegral(vector<Parameter>& r, vector<Parameter>& ed) {
	std::pair<double, double> res (0, 0);

	for(unsigned int i = 1; i < r.size(); i++) {
		double a = (r[i].value - r[i - 1].value) * (ed[i].value - ed[0].value);
		if(a < 0.0)
			res.second += -a;
		else
			res.first += a;
	}


	return res;
}