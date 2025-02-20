#include <vector>
#include <queue>
#include <iostream>
#include "csr_graph.h"
#include "io.cpp"

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
    for(int i = 0; i < g.t_vtx; i++){
        min_heap.push(i);
        if(min_heap.size() > 1000) min_heap.pop();
    }
    std::vector<int> top_vtx;
    while(!min_heap.empty()){
        int v = min_heap.top();
        min_heap.pop();
        std::cout << v << ": " << degrees[v] << std::endl;
        top_vtx.push_back(v);
    }
    return top_vtx;
}

std::vector<int> shortest_distances(const csr_graph& g, int start){
    std::vector<int> distances(g.t_vtx, -1);
    distances[start] = 0;
    std::vector<int> q(g.t_vtx);
    int front = 0, back = 0;
    q[back++] = start;
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

int main(int argc, char** argv){
    if(argc < 2){
        std::cerr << "Too few arguments provided" << std::endl;
        return -1;
    }
    csr_graph g = load_metis_graph(argv[1]);
    if(g.error) return -1;
    std::vector<int> top_vtx = top_degree(g);
    #pragma omp parallel for schedule(dynamic, 1)
    for(int i = 0; i < top_vtx.size(); i++){
        printf("Processing vtx %i: %i\n", i, top_vtx[i]);
        std::vector<int> distances = shortest_distances(g, top_vtx[i]);
    }
    return 0;
}