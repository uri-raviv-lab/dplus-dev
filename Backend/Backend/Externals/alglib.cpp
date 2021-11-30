/*************************************************************************
Copyright (c) 2005-2007, Sergey Bochkanov (ALGLIB project).

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

- Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

- Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer listed
  in this license in the documentation and/or other materials
  provided with the distribution.

- Neither the name of the copyright holders nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*************************************************************************/

#define _USE_MATH_DEFINES
#include <cmath>
#include <limits>

#include "alglib.h"

#include "Eigen/Core"
using Eigen::VectorXf;
using Eigen::VectorXd;

/*************************************************************************
Computation of nodes and weights for a Gauss-Legendre quadrature formula

The  algorithm  calculates  the  nodes  and  weights of the Gauss-Legendre
quadrature formula on domain [-1, 1].

Input parameters:
    n   –   a required number of nodes.
            n>=1.

	a   -   integration start

	b   -   integration end

Output parameters:
    x   -   array of nodes.
            Array whose index ranges from 0 to N-1.
    w   -   array of weighting coefficients.
            Array whose index ranges from 0 to N-1.

The algorithm was designed by using information from the QUADRULE library.
This algorithm was modified to use C++ and Eigen vectors by Tal Ben-Nun
*************************************************************************/
void buildgausslegendrequadrature(int n, double a, double b,
     VectorXd& x, VectorXd& w)
{
    double r;
    double r1;
    double p1;
    double p2;
    double p3;
    double dp3;
	const double epsilon = std::numeric_limits<double>::epsilon();

	x = VectorXd::Zero(n);
	w = VectorXd::Zero(n);
    for(int i = 0; i <= (n+1)/2-1; i++)
    {
        r = cos(M_PI*(4*i+3)/(4*n+2));
        do
        {
            p2 = 0;
            p3 = 1;
            for(int j = 0; j < n; j++)
            {
                p1 = p2;
                p2 = p3;
                p3 = ((2*j+1)*r*p2-j*p1)/(j+1);
            }
            dp3 = n*(r*p3-p2)/(r*r-1);
            r1 = r;
            r = r-p3/dp3;
        }
		while(fabs(r-r1)>= epsilon * (1+fabs(r))*100);
        x(i) = r;
        x(n-1-i) = -r;
        w(i) = 2/((1-r*r)*dp3*dp3);
        w(n-1-i) = 2/((1-r*r)*dp3*dp3);
    }

	for(int i = 0; i < n; i++)
    {
        x[i] = 0.5*(x[i]+1)*(b-a)+a;
        w[i] = 0.5*w[i]*(b-a);
    }
}

void buildgausslegendrequadrature(int n, float a, float b,
     VectorXf& x, VectorXf& w)
{
    float r;
    float r1;
    float p1;
    float p2;
    float p3;
    float dp3;
	const float epsilon = std::numeric_limits<float>::epsilon();

	x = VectorXf::Zero(n);
	w = VectorXf::Zero(n);
    for(int i = 0; i <= (n+1)/2-1; i++)
    {
        r = cosf(M_PI*float(4*i+3)/float(4*n+2));
        do
        {
            p2 = 0;
            p3 = 1;
            for(int j = 0; j < n; j++)
            {
                p1 = p2;
                p2 = p3;
                p3 = ((2*j+1)*r*p2-j*p1)/(j+1);
            }
            dp3 = n*(r*p3-p2)/(r*r-1);
            r1 = r;
            r = r-p3/dp3;
        }
		while(fabs(r-r1)>= epsilon * (1+fabs(r))*100);
        x(i) = r;
        x(n-1-i) = -r;
        w(i) = 2/((1-r*r)*dp3*dp3);
        w(n-1-i) = 2/((1-r*r)*dp3*dp3);
    }

	for(int i = 0; i < n; i++)
    {
        x[i] = 0.5f*(x[i]+1.0f)*(b-a)+a;
        w[i] = 0.5f*w[i]*(b-a);
    }
}
