#include <vector>
#include <cmath>
#include "csr_graph.h"

std::vector<float> normalize_columns(const csr_graph& g){
    std::vector<int> degrees(g.t_vtx, 0);
    // since the graph is undirected, the row size is equal to the corresponding column size
    for(int v = 0; v < g.t_vtx; v++){
        degrees[v] = g.row_map[v+1] - g.row_map[v];
    }
    std::vector<float> vals(g.nnz);
    for(int v = 0; v < g.t_vtx; v++){
        for(int j = g.row_map[v]; j < g.row_map[v+1]; j++){
            int u = g.entries[j];
            vals[j] = 1.0 / static_cast<float>(degrees[u]);
        }
    }
}

// sparse matrix-vector multiply
std::vector<float> spmv(const csr_graph& g, const std::vector<float>& vals, const std::vector<float>& input){
    std::vector<float> output(g.t_vtx, 0);
    for(int v = 0; v < g.t_vtx; v++){
        for(int j = g.row_map[v]; j < g.row_map[v+1]; j++){
            int u = g.entries[j];
            output[v] += input[u]*vals[j];
        }
    }
    return output;
}

void axpy(std::vector<float>& x, const float a, const float y){
    for(int i = 0; i < x.size(); i++){
        x[i] = a*x[i] + y;
    }
}

float norm(const std::vector<float>& a, const std::vector<float>& b){
    float x = 0;
    for(int i = 0; i < a.size(); i++){
        float diff = std::abs(a[i] - b[i]);
        x += diff*diff;
    }
    return std::sqrt(x);
}

std::vector<float> pagerank(const csr_graph& g, const float d, const float tol){
    float damp_add = (1.0 - d) / static_cast<float>(g.t_vtx);
    std::vector<float> transition_probs = normalize_columns(g);
    std::vector<float> pr(g.t_vtx, 1.0 / static_cast<float>(g.t_vtx));
    float diff = 100.0;
    while(diff > tol){
        std::vector<float> pr_next = spmv(g, transition_probs, pr);
        axpy(pr_next, d, damp_add);
        diff = norm(pr_next, pr);
        pr = pr_next;
    }
    return pr;
}