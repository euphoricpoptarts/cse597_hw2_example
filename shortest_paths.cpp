#include <vector>
#include <queue>
#include <fstream>
#include <iostream>

struct csr_graph {
    std::vector<int> row_map;
    std::vector<int> entries;
    int t_vtx;
    int nnz;
    bool error;
};

int fast_atoi( const char*& str )
{
    int val = 0;
    while(isdigit(*str)) {
        val = val*10 + static_cast<int>(*str - '0');
        str++;
    }
    return val;
}

void next_line(const char*& str){
    while(*str != '\n') str++;
    str++;
}

csr_graph load_metis_graph(const char *fname) {
    std::ifstream infp(fname, std::ios::binary);
    csr_graph g;
    if (!infp.is_open()) {
        std::cerr << "FATAL ERROR: Could not open metis graph file " << fname << std::endl;
        g.error = true;
        return g;
    }
    infp.seekg(0, std::ios::end);
    size_t sz = infp.tellg();
    std::cout << "Reading " << sz << " bytes from " << fname << std::endl;
    infp.seekg(0, std::ios::beg);
    //1 for extra newline if needed
    char* s = new char[sz + 1];
    infp.read(s, sz);
    infp.close();
    std::cout << "Finished reading bytes from " << fname << std::endl;
    //append an endline to end of file in case one doesn't exist
    //needed to prevent parser from overshooting end of buffer
    if(s[sz - 1] != '\n'){
        s[sz] = '\n';
        sz++;
    }
    const char* f = s;
    const char* fmax = s + sz;
    size_t header[4] = {0, 0, 0, 0};
    //ignore commented lines
    while(*f == '%') next_line(f);
    while(!isdigit(*f)) f++;
    //read header data
    for(int i = 0; i < 4; i++){
        header[i] = fast_atoi(f);
        while(!isdigit(*f)){
            if(*f == '\n'){
                i = 4;
                f++;
                break;
            }
            f++;
        }
    }
    int n = header[0];
    int m = header[1];
    int fmt = header[2];
    int ncon = header[3];
    bool has_ew = ((fmt % 10) == 1);
    if(fmt != 0 && fmt != 1){
        std::cerr << "FATAL ERROR: Unsupported format flags " << fmt << std::endl;
        std::cerr << "Graph parser does not currently support vertex weights" << std::endl;
        g.error = true;
        return g;
    }
    if(ncon != 0){
        std::cerr << "FATAL ERROR: Unsupported ncon " << ncon << std::endl;
        std::cerr << "Graph parser does not currently support vertex weights" << std::endl;
        g.error = true;
        return g;
    }
    std::vector<int> entries(m*2);
    std::vector<int> row_map(n + 1);
    row_map[0] = 0;
    int edges_read = 0;
    int rows_read = 0;
    //read edge information
    while(f < fmax){
        //increment past whitespace
        while(f < fmax && !isdigit(*f)){
            //ignore commented lines
            if(*f == '%'){
                next_line(f);
                continue;
            }
            if(*f == '\n'){
                //ignore extra trailing newlines
                if(rows_read < n) row_map[++rows_read] = edges_read;
            }
            f++;
        }
        if(f >= fmax) break;
        //fast_atoi also increments past numeric chars
        int edge_info = fast_atoi(f);
        //subtract 1 to convert to 0-indexed
        entries[edges_read] = edge_info - 1;
        edges_read++;
    }
    delete[] s;
    if(rows_read != n || edges_read != 2*m){
        std::cerr << "FATAL ERROR: Mismatch between expected and actual line/nonzero count in metis file" << std::endl;
        std::cerr << "Read " << rows_read << " lines and " << edges_read << " nonzeros" << std::endl;
        std::cerr << "Lines expected: " << n << "; Nonzeros expected: " << m*2 << std::endl;
        g.error = true;
        return g;
    }
    std::cout << "Processed bytes into graph" << std::endl;
    g.row_map = row_map;
    g.entries = entries;
    g.t_vtx = n;
    g.nnz = 2*m;
    g.error = false;
    return g;
}

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
    // #pragma omp parallel for schedule(dynamic, 2)
    for(int i = 0; i < top_vtx.size(); i++){
        printf("Processing vtx %i: %i\n", i, top_vtx[i]);
        std::vector<int> distances = shortest_distances(g, top_vtx[i]);
    }
    return 0;
}