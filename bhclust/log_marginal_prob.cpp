//<%
//cfg['compiler_args'] = ['-std=c++11']
//cfg['include_dirs'] = ['../eigen']
//setup_pybind11(cfg)
//%>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
#include <iostream>
#include <cmath>
#include <vector>
#include <numeric>

constexpr auto pi = 3.141592653589793;

double log_marginal_probability_cpp(Eigen::VectorXd m, Eigen::MatrixXd S,
    double r, double v, Eigen::MatrixXd X) {
    
    int N = X.rows();
    int p = X.cols();
    Eigen::VectorXd xsum = X.colwise().sum().transpose();

    Eigen::MatrixXd Sprime = S + X.transpose() * X + r * N / (N + r) * (m * m.transpose()) - 1 / (N + r) 
    * (xsum * xsum.transpose()) - r / (N + r) * (m * xsum.transpose() + xsum * m.transpose());

    double vprime = v + N;

    std::vector<double> gamma1(p, 0);
    std::vector<double> gamma2(p, 0);
    for (int i = 0; i < p; ++i) {
        gamma1[i] = lgamma((v - i) / 2);
        gamma2[i] = lgamma((vprime - i) / 2);
    }
    double gamma3 = std::accumulate(gamma1.begin(), gamma1.end(), 0.0);
	double gamma4 = std::accumulate(gamma2.begin(), gamma2.end(), 0.0);
	
	double log_prob = -N * p / 2 * log(2 * pi) + p / 2 * log(r / (N + r)) + v / 2 * log(S.determinant()) - vprime / 2
		* log(Sprime.determinant()) + vprime * p / 2 * log(2) + gamma4 - v * p / 2 * log(2) - gamma3;
	
	return log_prob;
}

PYBIND11_PLUGIN(log_marginal_prob) {
    pybind11::module m("log_marginal_prob", "auto-compiled c++ extension");
    m.def("log_marginal_probability_cpp", &log_marginal_probability_cpp);
    return m.ptr();
}