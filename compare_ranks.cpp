#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include "io.cpp"
#include "brandes_approx.cpp"
#include "pagerank.cpp"

// configures sort for decreasing order of score
class score_compare {
    std::vector<double>& score;
    
    public:
    score_compare(std::vector<double>& _score) : 
        score(_score) {}

    bool operator()(int a, int b) const {
        return score[a] > score[b];
    }
};

std::vector<int> get_rank(std::vector<double>& score){
    std::vector<int> vertices(score.size());
    // initialize vertices as 0, 1, 2, 3, ... n - 2, n - 1
    std::iota(vertices.begin(), vertices.end(), 0);
    score_compare comparator(score);
    std::sort(vertices.begin(), vertices.end(), comparator);
    std::vector<int> rank(score.size());
    // extract numeric ranking for each vertex
    for(int i = 0; i < score.size(); i++){
        int v = vertices[i];
        rank[v] = i;
    }
    return rank;
}

void spearman_correlation(std::vector<double>& a, std::vector<double>& b){
    std::vector<int> a_rank = get_rank(a);
    std::vector<int> b_rank = get_rank(b);
    double accumulate = 0;
    int n = a.size();
    for(int v = 0; v < n; v++){
        // cast to double to prevent overflow in diff*diff
        double diff = std::abs(a_rank[v] - b_rank[v]);
        accumulate += diff*diff;
    }
    // cast to double to prevent overflow
    double denom = static_cast<double>(n)*(static_cast<double>(n)*static_cast<double>(n) - 1.0);
    double corr = 1.0 - (6.0*accumulate / denom);
    std::cout << "Rank correlation: " << corr << std::endl;
}

int main(int argc, char** argv){
    if(argc < 2){
        std::cerr << "Too few arguments provided" << std::endl;
        return -1;
    }
    csr_graph g = load_metis_graph(argv[1]);
    if(g.error) return -1;
    std::vector<double> pr = pagerank(g, 0.85, 0.000000001);
    std::vector<double> b = approximate_betweenness(g);
    spearman_correlation(pr, b);
    return 0;
}