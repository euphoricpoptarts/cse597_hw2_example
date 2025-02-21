#include <vector>
#include <random>
#include "csr_graph.h"

std::vector<double> dependency_on_start(const csr_graph& g, int start){
    std::vector<int> distances(g.t_vtx, -1);
    std::vector<int> t_paths(g.t_vtx, 0);
    std::vector<double> dependence(g.t_vtx, 0);
    distances[start] = 0;
    t_paths[start] = 1;
    std::vector<int> q(g.t_vtx);
    int front = 0, back = 0;
    q[back++] = start;
    // compute distances and total paths from start to each vertex
    // using BFS
    while(front < back){
        int v = q[front++];
        int d = distances[v];
        for(int j = g.row_map[v]; j < g.row_map[v+1]; j++){
            int u = g.entries[j];
            if(distances[u] == -1){
                distances[u] = d + 1;
                q[back++] = u;
            }
            // if v is a predecessor of u
            if(distances[u] == d + 1){
                t_paths[u] += t_paths[v];
            }
        }
    }
    // compute dependency of start on each vertex
    // in reverse order of prior BFS traversal
    for(front = back - 1; front >= 0; front--){
        int v = q[front];
        int d = distances[v];
        double paths = t_paths[v];
        for(int j = g.row_map[v]; j < g.row_map[v+1]; j++){
            int w = g.entries[j];
            // if v is predecessor of w
            if(distances[w] == d + 1){
                dependence[v] += (paths / static_cast<double>(t_paths[w])) * (1.0 + dependence[w]);
            }
        }
    }
    return dependence;
}

std::vector<int> get_samples(int n, int sample_count){
    // standard boilerplate for uniform random distribution
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> uniform_vertex(0, n-1);
    int count = 0;
    std::vector<int> samples;
    while(count < sample_count){
        int x = uniform_vertex(gen);
        bool exists = false;
        // ensure duplicates are not inserted
        for(int y : samples){
            if(x == y){
                exists = true;
                break;
            }
        }
        if(!exists){
            samples.push_back(x);
            count++;
        }
    }
    return samples;
}

std::vector<double> approximate_betweenness(const csr_graph& g){
    std::vector<int> samples = get_samples(g.t_vtx, 20);
    double normalize = static_cast<double>(g.t_vtx) / static_cast<double>(samples.size());
    std::vector<double> b(g.t_vtx, 0);
    // accumulate pairwise dependences of samples on each vertex
    for(const int sample : samples){
        std::vector<double> dependence = dependency_on_start(g, sample);
        for(int v = 0; v < g.t_vtx; v++){
            if(sample != v) b[v] += dependence[v];
        }
    }
    for(int v = 0; v < g.t_vtx; v++){
        b[v] *= normalize;
    }
    return b;
}