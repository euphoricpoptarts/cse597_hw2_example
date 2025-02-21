#include <vector>
#include <cmath>
#include "csr_graph.h"

// normalize columns to sum to 1
std::vector<double> normalize_columns(const csr_graph& g){
    std::vector<int> degrees(g.t_vtx, 0);
    // since the graph is undirected, the row size is equal to the corresponding column size
    for(int v = 0; v < g.t_vtx; v++){
        degrees[v] = g.row_map[v+1] - g.row_map[v];
    }
    std::vector<double> vals(g.nnz);
    for(int v = 0; v < g.t_vtx; v++){
        for(int j = g.row_map[v]; j < g.row_map[v+1]; j++){
            int u = g.entries[j];
            vals[j] = 1.0 / static_cast<double>(degrees[u]);
        }
    }
    return vals;
}

// sparse matrix-vector multiply
std::vector<double> spmv(const csr_graph& g, const std::vector<double>& vals, const std::vector<double>& input){
    std::vector<double> output(g.t_vtx, 0);
    #pragma omp parallel for
    for(int v = 0; v < g.t_vtx; v++){
        for(int j = g.row_map[v]; j < g.row_map[v+1]; j++){
            int u = g.entries[j];
            output[v] += input[u]*vals[j];
        }
    }
    return output;
}

// a times x plus y
void axpy(std::vector<double>& x, const double a, const double y){
    for(int i = 0; i < x.size(); i++){
        x[i] = a*x[i] + y;
    }
}

// calculates 2-norm of vector difference
double norm2(const std::vector<double>& a, const std::vector<double>& b){
    double x = 0;
    for(int i = 0; i < a.size(); i++){
        double diff = std::abs(a[i] - b[i]);
        x += diff*diff;
    }
    return std::sqrt(x);
}

std::vector<double> pagerank(const csr_graph& g, const double d, const double tol){
    double damp_add = (1.0 - d) / static_cast<double>(g.t_vtx);
    std::vector<double> transition_probs = normalize_columns(g);
    std::vector<double> pr(g.t_vtx, 1.0 / static_cast<double>(g.t_vtx));
    double diff = 100.0;
    int iterations = 0;
    // convergence is measured by 2-norm of difference between successive iterations
    while(diff > tol){
        std::vector<double> pr_next = spmv(g, transition_probs, pr);
        axpy(pr_next, d, damp_add);
        diff = norm2(pr_next, pr);
        pr = pr_next;
        iterations++;
    }
    std::cout << "Iterations to converge: " << iterations << std::endl;
    return pr;
}