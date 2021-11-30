#pragma once 
#include "ParserSolver.h"
#include <cmath>
#include <limits>
#include <sstream>


string intToString(int number)
{
   stringstream ss;
   ss << number;
   return ss.str();
}
OPERATION ParserSolver::op(char o) {
	switch(o) {
		case '+':
			return ADD;
		case '-':
			return SUBTRACT;
		case '*':
			return MULTIPLY;
		case '/':
			return DIVIDE;
		default:
			(*error).push_back("Bad operation");
			return ADD;
	}
}

bool ParserSolver::isOperation(char c) {
	if(c == '-' || c == '+' || c == '/' || c == '*')
		return true;
	return false;
}

ParserSolver::ParserSolver(std::vector<std::string>& errs, std::string expres, ArrayXXd& dataX, ArrayXXd& dataY, ArrayXd& coefficients) {
	error = &errs;
	expressionE = expres;
	rawData = dataY;
	xdata = dataX;
	coefs = coefficients;
}

ParserSolver::~ParserSolver() {
}

ArrayXd ParserSolver::EvaluateExpression(ArrayXd& newX, std::string expres) {
	expressionE = expres;
	return EvaluateExpression(newX);
}

ArrayXd ParserSolver::EvaluateExpression(ArrayXd& newX) {
	(*error).clear();
	std::string expression = expressionE;
	ArrayXd res;

	// Crop the data to fit all used vectors
	if(!cropData()) {
		(*error).push_back("No overlapping x-range");
		return res;
	}
	newX = finalX;

	// Check to make sure that any parentheses are place correctly
	int j = 0;
	for(int i = 0; i < (int)expression.size(); i++) {
		char ci = expression.at(i);
		if(ci == '(' || ci == ')') {
			if(ci == '(') {
				j++;
			} else {
				j--;
				if(j < 0)
					(*error).push_back("Mismatched parentheses");
				if(i > 0 && expression.at(i - 1) == '(')
					(*error).push_back("Empty parentheses");
			} //else
		} // outer if parentheses
		
		if(!isalnum(ci) && !isOperation(ci) && !isspace(ci) && ci != '.' && ci != '(' && ci != ')')
			(*error).push_back("Illegal character");
	} // for

	if(j != 0)
		(*error).push_back("Mismatched number of parentheses");

	// Remove whitespace
	for(int i = expression.size() - 1; i >= 0; i--) {
		if(isspace(expression.at(i)))
			expression.erase(expression.begin() + i);
	}

	if(expression.size() == 0)
		(*error).push_back("Empty expression");
	else {
		char last = expression.at(expression.size() - 1);
		char first = expression.at(0);
		bool prevOp = false;

		if(isOperation(last))
			(*error).push_back("Expression ends in operation");
		if(first == '/' || first == '*')
			(*error).push_back("Expression starts with operation");
		for(int i = 0; i < (int)expression.size(); i++) {
			char cur = expression.at(i);
			if(prevOp && isOperation(cur)) {
				if(!(cur == '-' && (expression.at(i - 1) == '*' || expression.at(i - 1) == '/')))
					(*error).push_back("Two operations in a row");
			}
			prevOp = (isOperation(cur));
		}
		for(int i = 1; i < (int)expression.size(); i++) {
			last  = expression.at(i);
			first = expression.at(i-1);
			if(first == '(' && (isOperation(last) && last != '-'))
				(*error).push_back("Operation following left parentheses");
			if(last == ')' && isOperation(first))
				(*error).push_back("Operation precedes right parentheses");
			if((!isOperation(first) && last == '(' && first != '(')		||
				(!isOperation(last) && first == ')' && last != ')')		||
				(isalnum(first) && isalpha(last))						||
				(isalnum(last) && isalpha(first))
				)
				expression.insert(expression.begin() + i--, '*');
		}
	}

	if((*error).size() > 0)
		return res;

	expressionE = expression;	// Save the adjusted expression

	// Simplify expression for derivativations
	bool der = true;
	for(int i = 0; i < (int)expressionE.size(); i++)
		der &= (expressionE[i] != '/');
	dExpressionE = der ? simplifyExp(expressionE) : "";


	return CalculateExpression();
}

ArrayXd ParserSolver::CalculateExpression() {
	return CalculateExpression(expressionE);
}

ArrayXd ParserSolver::CalculateExpression(std::string expres) {
	std::string expression = expres;
	ArrayXd res;
	int startP = -1, endP = -1;
	int pos = 0, start = 0, cnt = 0, end = 0;
	bool inPar = false;

	// Deal with an initial '-' or '+'
	if(expression[0] == '-' || expression[0] == '+')
		expression.insert(0, "0");

	// First find and deal with all + and - not in parentheses (+ first, otherwise unwanted results can occur)
	cnt = 0;
	for(int i = 0; i < (int)expression.size(); i++) {
		if(expression[i] == '(')
			cnt++;
		if(expression[i] == ')')
			cnt--;
		inPar = cnt != 0;
		if(!inPar && expression[i] == '+') {
			std::string l = expression.substr(0, i), r = expression.substr(i+1, expression.size() - i);
			return CalculateExpression(l, r, op(expression[i]));
		}
	}
	cnt = 0;
	for(int i = 0; i < (int)expression.size(); i++) {
		if(expression[i] == '(')
			cnt++;
		if(expression[i] == ')')
			cnt--;
		inPar = cnt != 0;
		if(!inPar && (expression[i] == '-') && !isOperation(expression[i-1])) {
			std::string l = expression.substr(0, i), r = expression.substr(i+1, expression.size() - i);
			return CalculateExpression(l, r, op(expression[i]));
		}
	}

	// Next is *
	cnt = 0;
	for(int i = 0; i < (int)expression.size(); i++) {
		if(expression[i] == '(')
			cnt++;
		if(expression[i] == ')')
			cnt--;
		inPar = cnt != 0;
		if(expression[i] == '*' && !inPar) {
			std::string l = expression.substr(0, i), r = expression.substr(i+1, expression.size() - i);
			return CalculateExpression(l, r, op(expression[i]));
		}
	}
	
	// Last is /
	cnt = 0;
	for(int i = 0; i < (int)expression.size(); i++) {
		if(expression[i] == '(')
			cnt++;
		if(expression[i] == ')')
			cnt--;
		inPar = cnt != 0;
		if(expression[i] == '/' && !inPar) {
			std::string l = expression.substr(0, i), r = expression.substr(i+1, expression.size() - i);
			return CalculateExpression(l, r, op(expression[i]));
		}
	}

	// Deal with parentheses
	cnt = 0;
	for(int i = 0; i < (int)expression.size(); i++) {
		if(startP < 0) {	// First '('
			if(expression[i] == '(') {
				startP = i;
				cnt++;
			}
		} else { // Already found the first '('
			if(expression[i] == '(')
				cnt++;
			if(expression[i] == ')')
				cnt--;
			if(cnt == 0) { // Found matching ')'
				endP = i;
				break;
			}
		}	// if first '('
	}	// for

	if(endP > 0) {	// We have parentheses
		string paren = expression.substr(startP + 1, endP - startP - 1);
		
		// Special cases where the parentheses are at an end (or both)
		// Both
		if(startP == 0 && endP == expression.size() - 1)
			return CalculateExpression(paren);
		// End
		string l = expression.substr(0, startP - 1);
		if(endP == expression.size() - 1)
			return CalculateExpression(l, paren, op(expression[startP - 1]));
		// Beginning
		string r = expression.substr(endP + 1, expression.size() - endP);
		if(startP == 0)
			return CalculateExpression(paren, r, op(expression[endP + 1]));

		if(expression[startP-1] == '+' || expression[startP-1] == '-')	// Do the other first
			return opVector(CalculateExpression(l), op(expression[startP-1]), CalculateExpression(paren, r, op(expression[endP+1])));
		else
			return opVector(CalculateExpression(l, paren, op(expression[startP-1])), op(expression[endP+1]), CalculateExpression(r));
	}

	// If there are no operations... Determine if it's a number, lowercase (variable) or uppercase (curve)
	char *pEnd;
	double oneRes = strtod(expression.c_str(), &pEnd);
	if(pEnd != &expression[0]) {	// It's a number
		res = ArrayXd::Constant(finalX.size(), oneRes);	// TODO DEBUG: Size should be based on q
		return res;
	}

	// Check to make sure that it's one char
	if(expression.size() == 1) {
		if(isupper(expression[0])) {
			// This code works for a fixed 26 row data matrix. Change if needed to be more general (modify the entire class)
			res = data.col(int(expression[0] - 'A'));
			return res;
		//} else if(expression[0] == 'q') {	// Tal's rows; I don't like it as it screws up the (expression[0] - 'A')
		//	// Reserved code for q
		//	res = finalX;
		//	return res;
		} else {
			// This code works for a fixed 26 row data matrix. Change if needed to be more general (modify the entire class)
			res = ArrayXd::Constant(finalX.size(), coefs[int(expression[0] - 'a')]);
			return res;
		}

	} else
		(*error).push_back("Missing operation");

	// Should be empty
	return res;
}

ArrayXd ParserSolver::CalculateExpression(std::string left, std::string right, OPERATION op) {
	std::string l = left, r = right;
	return opVector(CalculateExpression(l), op, CalculateExpression(r));
}

ArrayXd ParserSolver::opVector(ArrayXd& left, OPERATION op, ArrayXd& right) {
	ArrayXd res;
	switch(op) {
		case ADD:
			return left + right;
		case SUBTRACT:
			return left - right;
		case MULTIPLY:
			return left * right;
		case DIVIDE:
			return left / right;
		default:
			(*error).push_back("Bad operation");
			return left;
	}
}

bool ParserSolver::cropData() {

	if((rawData.cols() != xdata.cols()) ||
		(rawData.rows() != xdata.rows()) ||
		(rawData.size() == 0)
		)
		return false;

	double small = 0.0, big = numeric_limits<double>::max();
	int noCurves	= rawData.cols(),
		len			= rawData.rows();
	
	if(xdata.cols() > coefs.size()) {
		small	= xdata(0, xdata.cols()-1);
		big		= xdata(1, xdata.cols()-1);
	}
	// Find the mutual range
	for(int i = 0; i < (int)expressionE.size(); i++) {
		char ci = expressionE[i];
		if(isupper(ci)) {
			double tmp = 0.0;
			// This code works for a fixed 26 row data matrix. Change if needed to be more general (modify the entire class)
			small = max(small, xdata(0, int(ci - 'A')));
			for(int j = 0; j < xdata.col(int(ci - 'A')).size(); j++)
				tmp = max(xdata(j, int(ci - 'A')), tmp);
				//if(xdata(j, int(ci - 'A')) > tmp)
				//	tmp = xdata(j, int(ci - 'A'));
			big = min(tmp, big);
			if(big <= small) {
				(*error).push_back("No mutual range");
				return false;
			} // if
		} // if
	} // for

	std::string check;
	// Create X vector of all points (finalX)
	std::vector<double> range;
	range.push_back(small);
	for(int i = 0; i < (int)expressionE.size(); i++) {
		char ci = expressionE[i];
		if(!isupper(ci))
			continue;
		// Already treated
		if(check.find(ci) != std::string::npos)
			continue;
		check += ci;
		int ii = int(ci - 'A');

		for(int j = 0; j < xdata.col(ii).size(); j++)
			range.push_back(xdata(j, ii));
	}
	std::sort(range.begin(), range.end());

	while(range.size() > 0 && range[0] < max(2.0 * numeric_limits<double>::epsilon(), small))
		range.erase(range.begin());
	while(range.size() > 0 && range[range.size() - 1] > big)
		range.erase(range.end() - 1);

	for(int i = 0; i < (int)range.size() - 1; i++)
		if(fabs(range[i]/range[i+1] - 1.0) < 1.0e-6)
			range.erase(range.begin() + 1 + i--);

	finalX.resize(range.size());
	for(int i = 0; i < (int)range.size(); i++)
		finalX(i) = range[i];


	// Interpolate all the rawData to data using the new X vector (data)
	data = ArrayXXd(finalX.size(), 26);
	check.clear();
	for(int i = 0; i < (int)expressionE.size(); i++) {
		char ci = expressionE[i];
		if(!isupper(ci))
			continue;
		// Already treated
		if(check.find(ci) != std::string::npos)
			continue;
		check += ci;
		int ii = int(ci - 'A');
		for(int j = 0; j < finalX.size(); j++)
			data(j, ii) = InterpolatePoint(finalX(j), xdata.col(ii), rawData.col(ii));
	} // for i

#ifdef _DEBUG
	std::vector<double> DEBX;
	for(int i = 0; i < finalX.size(); i++)
		DEBX.push_back(finalX(i));
#endif
	return finalX.size() >= 2;
} // cropData()

double ParserSolver::InterpolatePoint(double x0, const Eigen::ArrayXd &x, const Eigen::ArrayXd &y) {
	int st = 0, nd = x.size() - 1, mid, i;
#ifdef _DEBUG
	std::vector<double> DEB;
	for(int k = 0; k < x.size(); k++)
		DEB.push_back(x(k));
#endif
	// Ignore the trailing zeroes
	while((x(nd) <= x(nd - 1)) && (nd - st > 1)) {
		mid = (nd + st) / 2;
		if(x(mid) > x(nd))
			st = mid;
		else
			nd = mid;
	}

	if(nd - st <= 1) {
		int i = 0;
		for(i = st; i < nd; i++)
			if(x(i) >= x(i + 1))
				break;
		nd = i;
	}

	st = 0;
	// the ends
	if(fabs(x0/x(0) - 1.0) < 1.0e-6)
		return y(0);
	if(fabs(x0/x(nd) - 1.0) < 1.0e-6)
		return y(nd);

	if(x0 < x(0) || x0 > x(nd)) {
		(*error).push_back("Out of range (interpolate)");
		return 0.0;
	}

	do {
		mid = (nd + st) / 2;
		if(x0 > x[mid])
			st = mid + 1;
		else
			nd = mid;

	} while ((!(x0 >= x[mid] && x0 <= x[mid + 1]) && (st < nd)));

	i = mid + 1;
	if(x0 <= x[i] && i > 0)
		return y[i - 1] + (x0 - x[i - 1]) * ((y[i] - y[i - 1]) / (x[i] - x[i - 1]));

	(*error).push_back("Did not interpolate point");
	return 0.0;
}

ArrayXXd ParserSolver::GetMatrix() {
	if(cropData())
		return data;
	ArrayXXd grr;
	return grr;
}

void ParserSolver::ReplaceCoefficients(Eigen::ArrayXd &coefficients) {
	coefs = coefficients;
}

ArrayXd ParserSolver::InterpolateCurve(ArrayXd xIn, ArrayXd yIn) {
	ArrayXd yOut = ArrayXd::Zero(finalX.size());
	for(int i = 0; i < finalX.size(); i++)
		yOut(i) = InterpolatePoint(finalX(i), xIn, yIn);
	
	return yOut;
}

string ParserSolver::simplifyExp(string st) {
	int f = 0, l = 0, found = 0, per = 0, j = 0, sgn;
	string res = "", tmp = "";
	found = st.find("(");

	if(found == string::npos)
		return st;
	sgn = st.find_last_of("+-", found);
	if(sgn > 0)
		res += st.substr(0, sgn + (st[sgn] == '+' ? 1 : 0));
	//j = found;
	//do {
	//	if (st[j] == '(')
	//		per++;
	//	else if (st[j] == ')')
	//		per--;
	//	j++;
	//} while (per != 0 && st[j] != '-' && st[j] != '+' && j < (int)st.size() );

	string lef, rig, rrig, rest;
	breakUpString(st.substr(found, st.size() - found - 0), rig, rrig, rest);
	lef = st.substr(sgn + 1, max(0,found - sgn - 2));
	if(sgn >= 0 && st[sgn] == '-')
		lef = "-1" + lef;
	string tmpRes = multiplyExp(lef, rig);
	if(rrig.size() > 0)
		tmpRes = multiplyExp(tmpRes, rrig);
	if(rest.size() > 0) {
		tmp = simplifyExp(rest);
		if (rest[0] == '+' && tmp[0] != '-')
			tmpRes += '+';
		tmpRes += tmp;
	}
	res += tmpRes;

	return res;
}

string ParserSolver::multiplyExp(string l, string r) {
	l = simplifyExp(l);
	r = simplifyExp(r);
	if (l.size() == 0) 
		return r;
	else if (r.size() == 0)
		return l;

	string res = "";
	int prev = 0;

	vector<string> lS, rS;
	vector<bool> lOp, rOp;
	lOp.push_back(true);
	if(l[0] == '-') {
		lOp[0] = false;
		l.erase(l.begin());
	}

	for(int i = 0; i < (int)l.size(); i++) {
		if(l[i] == '+' || (l[i] == '-' && (i == 0 || !isOperation(l[i - 1]))) || (i == l.size() - 1)) {
			bool op = (l[i] == '+' || l[i] == '-');
			if(op)
				lOp.push_back(l[i] == '+');
			lS.push_back(l.substr(prev, i + (op ? 0 : 1) - prev));
			prev = i + 1;
			
		}
	}

	prev = 0;
	rOp.push_back(true);
	if(r[0] == '-') {
		rOp[0] = false;
		r.erase(r.begin());
	}

	for(int i = 0; i < (int)r.size(); i++) {
		if(r[i] == '+' || (r[i] == '-' && (i == 0 || !isOperation(r[i - 1]))) || (i == r.size() - 1)) {
			bool op = (r[i] == '+' || r[i] == '-');
			if(op)
				rOp.push_back(r[i] == '+');
			rS.push_back(r.substr(prev, i + (op ? 0 : 1) - prev));
			prev = i + 1;
		}
	}

	for(int i = 0; i < (int)lS.size(); i++) {
		for(int j = 0; j < (int)rS.size(); j++) {
			res += (lOp[i] ^ rOp[j]) ? "-" : "+";
			res += lS[i] + "*" + rS[j];
		}
	}

	if(res[0] == '+')
		res.erase(res.begin());
	if (res[res.size()-1] == '+' || res[res.size()-1] == '-')
		res.erase(res.end());
	return res;
}

void ParserSolver::breakUpString(const string str, string &r, string &rr, string &rest) {
	int cnt = 0, blanks = 0;
	int i = 0;
	for(; i < (int)str.size(); i++) {
		if (str[i] == '(') 
			cnt++;
		else if (str[i] == ')') 
			cnt--;
		if(cnt == 0) {
			r = str.substr(1, i++ - 1);
			break;
		}
	}

	if(i < (int)str.size() && str[i] == '*')
		blanks = 1;

	for(; i < (int)str.size(); i++) {
		if (str[i] == '(') 
			cnt++;
		else if (str[i] == ')') 
			cnt--;
		if (cnt == 0 && (str[i] == '-' || str[i] == '+')) {
			rr = str.substr(r.size() + 2 + blanks, max(0 ,i - ((int)r.size() + 2 + blanks)));
			break;
		}
		else if (cnt == 0 && i == str.size() - 1){
			rr = str.substr(r.size() + 2 + blanks, max(0 ,i - ((int)r.size() + 1 + blanks)));
			break;
		}
	}
	if (i != str.size() - 1)
		rest = str.substr(i, str.size() - i);
	else 
		rest = "";
}

bool ParserSolver::CanExactDerive() {
	return dExpressionE.size() > 0;
}

ArrayXd ParserSolver::Derive(char ch) {
	string tmp = expressionE;
	string dExp = dExpressionE;
	int f = 0, count;
	// Change expressionE to the derivative
	for(int i = 0; i < (int)dExp.size(); i++) {
		count = 0;
		if (i == dExp.size() - 1 ||  dExp[i] == '+' || dExp[i] == '-') {
			for (int j = f; j < i; j++) {
				if (dExp[j] == ch)
					count++;
			}
			if (count>0)
				dExp.replace(dExp.find(ch, f), 1, (intToString(count)));
			else {
				dExp.erase(max(0, f - 1), i - f + min(f, (i == (int)dExp.size() - 1 ? 2 : 1 )));
				i = f - min(f, 1);
			}
			f = i + (intToString(count).size());
		}
	}
	ArrayXd res = CalculateExpression(dExp);
	expressionE = tmp;
	return res;
}
