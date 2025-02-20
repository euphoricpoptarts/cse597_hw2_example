#include <vector>
#include <cmath>
#include "csr_graph.h"
#include "io.cpp"

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

void axpy(std::vector<double>& x, const double a, const double y){
    for(int i = 0; i < x.size(); i++){
        x[i] = a*x[i] + y;
    }
}

double norm(const std::vector<double>& a, const std::vector<double>& b){
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
    while(diff > tol){
        std::vector<double> pr_next = spmv(g, transition_probs, pr);
        axpy(pr_next, d, damp_add);
        diff = norm(pr_next, pr);
        std::cout << "Normalized diff: " << diff << std::endl;
        pr = pr_next;
    }
    return pr;
}

int main(int argc, char** argv){
    if(argc < 2){
        std::cerr << "Too few arguments provided" << std::endl;
        return -1;
    }
    csr_graph g = load_metis_graph(argv[1]);
    if(g.error) return -1;
    std::vector<double> rank = pagerank(g, 0.85, 0.000000001);
    return 0;
}