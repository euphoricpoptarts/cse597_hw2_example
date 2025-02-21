#include <vector>
#include <queue>
#include <iostream>
#include "csr_graph.h"
#include "io.cpp"

// configures priority-queue as a min-heap based on degree
class degree_compare {
    std::vector<int>& degrees;
    
    public:
    degree_compare(std::vector<int>& _degrees) : 
        degrees(_degrees) {}

    bool operator()(int a, int b) const {
        return degrees[a] > degrees[b];
    }
};

std::vector<int> top_degree(const csr_graph& g){
    std::vector<int> degrees(g.t_vtx);
    for(int i = 0; i < g.t_vtx; i++){
        degrees[i] = g.row_map[i+1] - g.row_map[i];
    }
    degree_compare comparator(degrees);
    std::priority_queue<int, std::vector<int>, degree_compare> min_heap(comparator);
    // maintain the top 1000 vertices by degree in the heap
    for(int i = 0; i < g.t_vtx; i++){
        min_heap.push(i);
        // drop the smallest degree vertex in heap if size exceeds 1000
        if(min_heap.size() > 1000) min_heap.pop();
    }
    std::vector<int> top_vtx;
    while(!min_heap.empty()){
        int v = min_heap.top();
        min_heap.pop();
        top_vtx.push_back(v);
    }
    return top_vtx;
}

std::vector<int> shortest_distances(const csr_graph& g, int source){
    std::vector<int> distances(g.t_vtx, -1);
    distances[source] = 0;
    std::vector<int> q(g.t_vtx);
    int front = 0, back = 0;
    q[back++] = source;
    // compute distances from source with BFS
    while(front < back){
        int v = q[front++];
        int d = distances[v];
        for(int j = g.row_map[v]; j < g.row_map[v+1]; j++){
            int u = g.entries[j];
            if(distances[u] == -1){
                distances[u] = d + 1;
                q[back++] = u;
            }
        }
    }
    return distances;
}

double closeness_centrality(const std::vector<int>& distances, int source){
    int n = distances.size();
    double t_distance = 0;
    for(int i = 0; i < n; i++){
        if(i == source) continue;
        double d = distances[i];
        // not sure how to handle this case
        if(d == -1) continue;
        t_distance += d;
    }
    double c = n - 1;
    c /= t_distance;
    return c;
}

double harmonic_centrality(const std::vector<int>& distances, int source){
    int n = distances.size();
    double h = 0;
    for(int i = 0; i < n; i++){
        if(i == source) continue;
        double d = distances[i];
        if(d == -1) continue;
        h += 1.0 / d;
    }
    // normalize
    h /= static_cast<double>(n - 1);
    return h;
}

int main(int argc, char** argv){
    if(argc < 2){
        std::cerr << "Too few arguments provided" << std::endl;
        return -1;
    }
    csr_graph g = load_metis_graph(argv[1]);
    if(g.error) return -1;
    std::vector<int> top_vtx = top_degree(g);
    std::vector<double> hc(top_vtx.size());
    std::vector<double> cc(top_vtx.size());
    #pragma omp parallel for schedule(dynamic, 1)
    for(int i = 0; i < top_vtx.size(); i++){
        printf("Processing vtx %i: %i\n", i, top_vtx[i]);
        std::vector<int> distances = shortest_distances(g, top_vtx[i]);
        hc[i] = harmonic_centrality(distances, top_vtx[i]);
        cc[i] = closeness_centrality(distances, top_vtx[i]);
    }
    for(int i = 0; i < top_vtx.size(); i++){
        int v = top_vtx[i];
        std::cout << "Vtx " << v << " has closeness centrality " << cc[i];
        std::cout << " and harmonic centrality " << hc[i] << std::endl;
    }
    return 0;
}