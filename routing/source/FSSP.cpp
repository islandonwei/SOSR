#include "h2h.h"
#include "spf.h"
#include <iostream>
#include <string>
#include <cstdlib>
#include <ctime>
#include "EmbeddingService.h"
#include "safe_geometry.h"

int nofcats, K;
// Both pruning rules are enabled for FSSP by default.  They can be disabled
// independently for the paper's pruning ablation through CLI flags.
bool enableBoundsPruning = true;
bool enableSPFPruning = true;
bool buildSPFIndices = false;
bool verifyExact = false;
bool spfAllCategories = false;
bool spfLegacyCategoryFilter = false;
string hsmCandidateFile;
int spfCandidateSize = 4;
int spfTopk = 2;
int spfMinCategorySize = 1;
unsigned int spfSeed = 20260906;
vector<vector<double>> categoryEmbeddings; //the first one is empty
vector<string> catname; //the first one is empty
void readCategoryID(string sfile) {
    string prefix = string("../data/") + sfile + string("/");
    string categoryIDFile = prefix + string("Categories_ID.txt");
    ifstream file(categoryIDFile);
    // The published toy/running-example data uses a lower-case filename,
    // whereas the four real datasets use the upper-case variant.
    if (!file) {
        categoryIDFile = prefix + string("categories_ID.txt");
        file.open(categoryIDFile);
    }
    if (!file) {
        throw runtime_error("Cannot open category ID file under " + prefix);
    }
    string line;
    nofcats = 0;
    catname.push_back(string(""));
    while (getline(file, line)) {
        if (line.empty()) continue;  
        int pos = line.find(' ');
        stringstream ss(line);
        int id;
        ss >> id;
        catname.push_back(line.substr(pos + 1));
        nofcats++;
    }
    if (nofcats == 0) throw runtime_error("Category ID file is empty: " + categoryIDFile);
    cout << "nofcats " << nofcats << endl;
}
vector<string> POInode[MAX_V];

vector<vector<int>> readHSMCandidateFile(const string& filename, int requirements) {
    ifstream input(filename);
    if (!input) throw runtime_error("Cannot open HSM candidate file: " + filename);
    vector<vector<int>> result;
    string line;
    while (getline(input, line)) {
        stringstream stream(line);
        vector<int> categories;
        int category;
        while (stream >> category) {
            if (category < 1 || category > nofcats)
                throw runtime_error("HSM candidate ID out of range: " + to_string(category));
            categories.push_back(category);
        }
        if (categories.empty()) throw runtime_error("HSM candidate file has an empty requirement line");
        sort(categories.begin(), categories.end());
        categories.erase(unique(categories.begin(), categories.end()), categories.end());
        result.push_back(categories);
    }
    if (static_cast<int>(result.size()) != requirements)
        throw runtime_error("HSM candidate count does not match requirement count");
    return result;
}
void readPOINodeMapping(string sfile){
    string prefix = string("../data/") + sfile + string("/");
    ifstream in(prefix + string("poi_node_mapping.txt"));
    string line;
    int node = -1;
    set<int> cats, allcats;
    map<int, set<int>> categoryToNodes;
    
    if (!in) throw runtime_error("Cannot open POI-node mapping under " + prefix);
    while (getline(in, line)) {
        if (line.find('-') != string::npos) {
            if (node != -1) {
                for (int c : cats) {
                    categoryToNodes[c].insert(node-1);
                }
            }
            node = stoi(line.substr(0, line.find('-')));
            cats.clear();
        } else {
            int pos = line.find('<');
            POInode[node - 1].push_back(line.substr(0, pos));
            if (pos != string::npos) {
                stringstream ss(line.substr(pos + 1));
                string cat;
                while (getline(ss, cat, '^')) {
                    if (!cat.empty()){
                        cats.insert(stoi(cat));
                        allcats.insert(stoi(cat));
                    }
                }
            }
        }
    }
    
    if (node != -1) {
        for (int c : cats) {
            categoryToNodes[c].insert(node-1);
        }
    }
    for (int c : allcats){
        for(int j: categoryToNodes[c]){
            category2Nodes[c].push_back(j);
        }
    }
    in.close();
}

void readCategoryEmbeddings(string sfile) {
    string prefix = string("../data/") + sfile + string("/");
    string categoryEmbeddingFile = prefix + string("category_embeddings.txt");
    
    ifstream file(categoryEmbeddingFile);
    if (!file) throw runtime_error("Cannot open category embeddings: " + categoryEmbeddingFile);
    categoryEmbeddings.clear();
    vector<double> emb1;
    categoryEmbeddings.push_back(emb1);
    string line;

    while (getline(file, line)) {
        if (line.empty()) continue;  
        vector<double> embedding;
        stringstream ss(line);
        double value;
        while (ss >> value) {
            embedding.push_back(value);
        }
        if (!embedding.empty()) {
            categoryEmbeddings.push_back(embedding);
        }
    }
    
    file.close();
    if (categoryEmbeddings.size() != static_cast<size_t>(nofcats + 1)) {
        throw runtime_error("Category embedding count does not match Categories_ID.txt");
    }
    cout << "Loaded " << categoryEmbeddings.size() - 1 << " category embeddings, dimension: " 
         << (categoryEmbeddings.empty() ? 0 : categoryEmbeddings[1].size()) << endl;
}


double dotProduct(const vector<double>& a, const vector<double>& b) {
    if (a.size() != b.size()) return -1.0;
    
    double result = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        result += a[i] * b[i];
    }
    return result;
}

void findSimilarCategories(const vector<double>& queryEmbedding, double alpha, vector<int>& result) {
    result.clear();
    double top1 = -1;
    for (size_t i = 1; i <= categoryEmbeddings.size() - 1; ++i) {
        double similarity = dotProduct(queryEmbedding, categoryEmbeddings[i]);
        top1 = max(top1, similarity);
    }
    //cout << top1 << " " << alpha << " " << categoryEmbeddings.size() - 1 << endl;
    for (size_t i = 1; i <= categoryEmbeddings.size() - 1; ++i) {
        double similarity = dotProduct(queryEmbedding, categoryEmbeddings[i]);
        //cout << catname[i] << "---" << similarity << endl;
        if (similarity >= top1 - alpha) {
            result.push_back((int)(i));
        }
    }   
}

void selectRepresentativePoints(const vector<int>& pointSet, int maxRepPoints, double& gridSize, vector<int>& representatives) {
    representatives.clear();
    if (pointSet.empty()) {
        //cout << "pointSet empty!" << endl;
        return;
    }
    
    if ((int)pointSet.size() <= maxRepPoints) {
        for(auto a: pointSet)
            representatives.push_back(a);
        gridSize = 0;
        return;
    }
    //srand(time(nullptr));
    // 二分搜索合适的gridSize
    double minGridSize = 0.001;
    double maxGridSize = 1.0;
    gridSize = (minGridSize + maxGridSize) / 2.0;

    while (maxGridSize - minGridSize > 0.0001) {
        map<pair<int, int>, vector<int>> grid;
        
        for (int nodeId : pointSet) {
            double x = coords[nodeId].first;
            double y = coords[nodeId].second;
            int gridX = (int)(x / gridSize);
            int gridY = (int)(y / gridSize);
            grid[make_pair(gridX, gridY)].push_back(nodeId);
        }
        
        int repCount = grid.size();
        
        if (abs(repCount - maxRepPoints) <= 10) {
            representatives.clear();
            for (auto& entry : grid) {
                vector<int>& gridPoints = entry.second;
                int randomIndex = rand() % gridPoints.size();
                representatives.push_back(gridPoints[randomIndex]);
            }
            return;
        }
        
        if (repCount > maxRepPoints) {
            minGridSize = gridSize;
            gridSize = (gridSize + maxGridSize) / 2.0;
        } else {
            maxGridSize = gridSize;
            gridSize = (minGridSize + gridSize) / 2.0;
        }
    }
    
    if (representatives.empty()) {
        map<pair<int, int>, vector<int>> grid;
        
        for (int nodeId : pointSet) {
            double x = coords[nodeId].first;
            double y = coords[nodeId].second;
            int gridX = (int)(x / gridSize);
            int gridY = (int)(y / gridSize);
            grid[make_pair(gridX, gridY)].push_back(nodeId);
        }
        
        for (auto& entry : grid) {
            vector<int>& gridPoints = entry.second;
            int randomIndex = rand() % gridPoints.size();
            representatives.push_back(gridPoints[randomIndex]);
        }
    }
}

vector<vector<int>> categoryRepresentatives;
void buildCategoryRepresentatives(int maxRep) {
    //FILE *fp_representatives = fopen("CategoryRepresentatives.txt", "w");
    vector<int> tmp1;
    categoryRepresentatives.push_back(tmp1);
    for (int i = 1; i <= nofcats; ++i) {
        double gridSize;
        vector<int> tmp;
        selectRepresentativePoints(category2Nodes[i], maxRep, gridSize, tmp);
        categoryRepresentatives.push_back(tmp);
        //fprintf(fp_representatives, "GridLen %f | # of representatives %d\n", gridSize, categoryRepresentatives[i].size());
    }
    cout << "\nBuilt representatives for " << categoryRepresentatives.size() - 1 << " categories" << endl;
    //fclose(fp_representatives);
}

struct Grid {
    double minX, maxX, minY, maxY;  
    vector<int> nodeIds;          
};

vector<DD> planeCoords;//lon lat to plane
map<int, vector<Grid>> category2Grids;
safe_geometry::Certificate geometricCertificate;
uint64_t geometricExpandedRectangles = 0;
double geometricCalibrationMilliseconds = 0.0;
void buildCategoryGrids(int maxNumGrids) {
    geometricCertificate = {};
    geometricExpandedRectangles = 0;
    double MPI = 3.1415926;
    double totalLat = 0.0;
    for (const auto& coord : coords) {
        totalLat += coord.second;
    }
    double avg_lat = totalLat / coords.size();
    double avg_lat_rad = avg_lat * MPI / 180.0;
    double lat_factor = cos(avg_lat_rad);
    
    planeCoords.resize(coords.size());
    for (size_t i = 0; i < coords.size(); ++i) {
        planeCoords[i].first = coords[i].first * lat_factor * 111320; 
        planeCoords[i].second = coords[i].second * 111320;           
        //printf("%d %d %d^^", i, planeCoords[i].first, planeCoords[i].second);
    }
    
    for (auto& [catId, nodeList] : category2Nodes) {
        if (nodeList.empty()) continue;
        
        double minX = planeCoords[nodeList[0]].first;
        double maxX = planeCoords[nodeList[0]].first;
        double minY = planeCoords[nodeList[0]].second;
        double maxY = planeCoords[nodeList[0]].second;
        
        for (int nodeId : nodeList) {
            double x = planeCoords[nodeId].first;
            double y = planeCoords[nodeId].second;
            minX = min(minX, x);
            maxX = max(maxX, x);
            minY = min(minY, y);
            maxY = max(maxY, y);
        }
        
        double width = maxX - minX;
        double height = maxY - minY;
        
        if (width == 0) width = 100;  
        if (height == 0) height = 100;
        
        // 二分搜索合适的gridSize
        double left = 1.0, right = max(width, height);
        double gridSize = right;
        
        while (right - left > 0.1) {
            double mid = (left + right) / 2;
            int gridCountX = (int)ceil(width / mid);
            int gridCountY = (int)ceil(height / mid);
            int totalGrids = gridCountX * gridCountY;
            
            if (abs(totalGrids - maxNumGrids) <= 50) {
                gridSize = mid;
                break;
            }
            
            if (totalGrids > maxNumGrids) {
                left = mid;  
            } else {
                right = mid; 
            }
        }
        
        map<pair<int, int>, vector<int>> gridMap;
        for (int nodeId : nodeList) {
            double x = planeCoords[nodeId].first;
            double y = planeCoords[nodeId].second;
            int gridX = (int)((x - minX) / gridSize);
            int gridY = (int)((y - minY) / gridSize);
            gridMap[{gridX, gridY}].push_back(nodeId);
        }
        
        vector<Grid> grids;
        for (auto& [gridPos, nodes] : gridMap) {
            Grid grid;
            grid.minX = minX + gridPos.first * gridSize;
            grid.maxX = minX + (gridPos.first + 1) * gridSize;
            grid.minY = minY + gridPos.second * gridSize;
            grid.maxY = minY + (gridPos.second + 1) * gridSize;
            grid.nodeIds = nodes;
            // Only expand rounded boundaries when required to contain an
            // assigned vertex. Candidate membership/grid partition is unchanged.
            const auto oldBox = make_tuple(grid.minX, grid.maxX, grid.minY, grid.maxY);
            for (int node : nodes) {
                grid.minX = min(grid.minX, planeCoords[node].first);
                grid.maxX = max(grid.maxX, planeCoords[node].first);
                grid.minY = min(grid.minY, planeCoords[node].second);
                grid.maxY = max(grid.maxY, planeCoords[node].second);
            }
            geometricExpandedRectangles += oldBox != make_tuple(grid.minX, grid.maxX, grid.minY, grid.maxY);
            grids.push_back(grid);
        }
        
        category2Grids[catId] = grids;
        //cout << "Category " << catId << ": " << nodeList.size() << " nodes -> " << grids.size() << " grids" << endl;
    }
    
    const auto geometricCalibrationStart = chrono::steady_clock::now();
    geometricCertificate = safe_geometry::calibrate(planeCoords, G, N);
    geometricCalibrationMilliseconds = chrono::duration<double, milli>(chrono::steady_clock::now() - geometricCalibrationStart).count();
    cout << "Built grids for " << category2Grids.size() << " categories" << endl;
    cout << setprecision(17) << "GEOMETRY_CERTIFICATE scale=" << geometricCertificate.scale
         << " arcs=" << geometricCertificate.arcs
         << " verified_violations=" << geometricCertificate.verification_violations
         << " expanded_rectangles=" << geometricExpandedRectangles << endl;
}


double pointToRectDistance(double px, double py, double minX, double maxX, double minY, double maxY) {
    double dx = max(0.0, max(minX - px, px - maxX));
    double dy = max(0.0, max(minY - py, py - maxY));
    return sqrt(dx * dx + dy * dy);
}

vector<vector<int>> Vr;
void buildVrWithPruning(int s, int t, double wub, vector<vector<int>>& similarCategories) {
    if (enableBoundsPruning && (!geometricCertificate.ready ||
        geometricCertificate.nodes != planeCoords.size()))
        throw runtime_error("Missing or stale geometric lower-bound certificate");
    
    int K = similarCategories.size();
    
    double s_x = planeCoords[s].first;
    double s_y = planeCoords[s].second;
    double t_x = planeCoords[t].first;
    double t_y = planeCoords[t].second;

    //printf("%f %f %f %f %f\n", s_x, s_y, t_x, t_y, wub);

    for (int i = 0; i < K; ++i) {
        set<int> uniqueNodesVr;
        int totalGrids = 0;
        int prunedGrids = 0;
        
        for (int catId : similarCategories[i]) {
            if (category2Grids.find(catId) == category2Grids.end()) {
                continue;
            }
            
            for (const Grid& grid : category2Grids[catId]) {
                totalGrids++;
                
                double distS = safe_geometry::rectangleDistanceLower(s_x, s_y, grid.minX, grid.maxX, grid.minY, grid.maxY);
                double distT = safe_geometry::rectangleDistanceLower(t_x, t_y, grid.minX, grid.maxX, grid.minY, grid.maxY);
                double totalDist = enableBoundsPruning ? safe_geometry::routeLower(geometricCertificate, distS, distT) : 0.0;
                
                if (enableBoundsPruning && safe_geometry::exceedsUpper(totalDist, wub)) {
                    prunedGrids++;
                    continue;
                }
                
                for (int nodeId : grid.nodeIds) {
                    uniqueNodesVr.insert(nodeId);
                }
            }
        }
        
        Vr[i] = vector<int>(uniqueNodesVr.begin(), uniqueNodesVr.end());
        //cout << "Requirement " << i << ": " << totalGrids << " grids, pruned " << prunedGrids << " grids, final Vr size: " << Vr[i].size() << endl;
    }
}

double getUpperBound(int s, int t, vector<vector<double>> &queryEmbeddings, vector<vector<int>> &similarCategories, vector<int> &vub, vector<double> &dub) {    
    vector<vector<int>> candidateSets;
    candidateSets.resize(K);
    for (int i = 0; i < K; ++i) {
        set<int> uniqueNodes;
        for (int catId : similarCategories[i]) {
            if (catId < (int)categoryRepresentatives.size()) {
                for (int nodeId : categoryRepresentatives[catId]) {
                    uniqueNodes.insert(nodeId);
                }
            }
        }
        candidateSets[i] = vector<int>(uniqueNodes.begin(), uniqueNodes.end());
        if (candidateSets[i].empty()) {
            throw runtime_error("No representative candidate for requirement " + to_string(i));
        }
        /*cout << "Requirement " << i << " has " << candidateSets[i].size() << " representatives" << endl;
        for(int c:candidateSets[i])
            cout << c+1 << " ";
        cout << endl;*/
    }

    vector<map<int, double>> dp(K + 1);  
    vector<map<int, int>> parent(K + 1); 
    dp[0][s] = 0.0;

    for (int i = 1; i <= K; ++i) {
        for (int v : candidateSets[i-1]) {
            dp[i][v] = DBL_MAX;
        }
        
        for (auto& [u, dist_u] : dp[i-1]) {
            for (int v : candidateSets[i-1]) {
                double newDist = dist_u + h2hQuery(u, v);
                if (newDist < dp[i][v]) {
                    dp[i][v] = newDist;
                    parent[i][v] = u;
                }
            }
        }
    }

    double minDist = DBL_MAX;
    int bestEndNode = -1;
    
    for (auto& [v, dist_v] : dp[K]) {
        double totalDist = dist_v + h2hQuery(v, t);
        if (totalDist < minDist) {
            minDist = totalDist;
            bestEndNode = v;
        }
    }

    if (bestEndNode != -1) {
        int currentNode = bestEndNode;
        double rdis = h2hQuery(currentNode, t);
        for (int i = K; i >= 1; --i) {
            vub.push_back(currentNode);
            //dub.push_back(minDist - dp[i][currentNode]);
            dub.push_back(rdis);
            rdis += h2hQuery(currentNode, parent[i][currentNode]);
            currentNode = parent[i][currentNode];
        }
    }
    //reverse(path.begin(), path.end());
    return minDist;
}
uint64_t prunedwub, prunedspf, nh2hcalls;
double FSSPQuery(int s, int t, vector<vector<double>> &queryEmbeddings, vector<vector<int>> &similarCategories, vector<int> &path) {
    // FSSP operates on candidate sets, not embeddings.  With an external
    // HSM candidate file queryEmbeddings is intentionally empty.
    int k = similarCategories.size();
    path.clear();
    Vr.resize(k);
    vector<int> vub;
    vector<double> dub;    
    double wub = getUpperBound(s, t, queryEmbeddings, similarCategories, vub, dub);
    //cout << "wub " << wub << endl;

    // buildVrWithPruning(s, t, DBL_MAX, similarCategories);
    buildVrWithPruning(s, t, wub, similarCategories);
    for (int i = 0; i < k; ++i) {
        if (Vr[i].empty()) {
            throw runtime_error("All candidates were pruned for requirement " + to_string(i));
        }
    }

    vector<map<int, double>> dp(k + 1);
    vector<map<int, int>> parent(k + 1);
    dp[0][s] = 0.0;
    for (int i = 1; i <= k; ++i) {
        for (int v : Vr[i-1]) {
            dp[i][v] = DBL_MAX;
        }
        
        for (auto& [u, dist_u] : dp[i-1]) {
            if (enableBoundsPruning && i > 1){
                wub = min(wub, dist_u + h2hQuery(u, vub[k-i]) + dub[k-i]);
                if (dist_u + h2hQuery(u, t) - wub>1e-7) {
                    prunedwub++;
                    continue;
                }
            }
            int tmps = 0;
            const vector<int> pruned = enableSPFPruning ?
                applySPFPruning(u, Vr[i - 1], similarCategories[i-1]) : Vr[i - 1];
            for (int v : pruned)
            {
                tmps++;
                nh2hcalls++;
                double newDist = dist_u + h2hQuery(u, v);
                if (1e-8 <= dp[i][v]-newDist) {
                    dp[i][v] = newDist;
                    parent[i][v] = u;
                }
            }
            if (enableSPFPruning) prunedspf += Vr[i - 1].size() - tmps;
        }
    }

    double minDist = DBL_MAX;
    int bestEndNode = -1;
    
    for (auto& [v, dist_v] : dp[k]) {
        double totalDist = dist_v + h2hQuery(v, t);
        if (totalDist < minDist) {
            minDist = totalDist;
            bestEndNode = v;
        }
    }
    
    path.push_back(t);
    if (bestEndNode != -1) {
        int currentNode = bestEndNode;
        for (int i = k; i >= 1; --i) {
            path.push_back(currentNode);
            currentNode = parent[i][currentNode];
        }
        path.push_back(currentNode);
    }
    reverse(path.begin(), path.end());
    return minDist;
}

// Exact DP is intentionally limited to the verification harness.  It shares
// the same candidate construction as FSSP but does not apply either pruning
// rule, so equality validates the safety of the implementation on a sampled
// workload without changing the timed algorithm.
double exactDPQuery(int s, int t, const vector<vector<int>>& similarCategories) {
    vector<map<int, double>> dp(similarCategories.size() + 1);
    dp[0][s] = 0.0;
    for (size_t i = 1; i <= similarCategories.size(); ++i) {
        set<int> candidates;
        for (int category : similarCategories[i - 1]) {
            for (int node : category2Nodes[category]) candidates.insert(node);
        }
        if (candidates.empty()) throw runtime_error("Exact DP has no candidate vertices");
        for (int v : candidates) {
            double best = DBL_MAX;
            for (const auto& [u, prior] : dp[i - 1]) best = min(best, prior + h2hQuery(u, v));
            dp[i][v] = best;
        }
    }
    double best = DBL_MAX;
    for (const auto& [v, prior] : dp.back()) best = min(best, prior + h2hQuery(v, t));
    return best;
}

EmbeddingService* g_embeddingService = nullptr;
int initTransformer(){
    try {
        std::cout << "Initializing embedding service..." << std::endl;
        g_embeddingService = new EmbeddingService();
        return 0; 
        
    } catch (const std::exception& e) {
        std::cerr << "Error in initTransformer: " << e.what() << std::endl;
        return -1;
    }
}

vector<string> poi_names;
int main(int argc, char* argv[]) {
    string sfile, reqfile;
    int candidate_size, topk, catid, startqi, endqi = 5, maxQueries = -1;
    double alpha;
    if (argc > 1) {
        sfile = string(argv[1]);
    } else {
        sfile = string("NYC");
    }
    if (argc > 2) {
        reqfile = string(argv[2]);
        startqi = 4;
    } else {
        reqfile = string("req3");
        startqi = 0;
    }
    if (argc > 3) {
        alpha = stod(string(argv[3]));
    } else {
        alpha = 0.05;
    }
    for (int argi = 4; argi < argc; ++argi) {
        string option(argv[argi]);
        if (option == "--all-q") {
            startqi = 0;
        } else if (option == "--q-index" && argi + 1 < argc) {
            startqi = stoi(argv[++argi]) - 1;
            if (startqi < 0 || startqi >= endqi)
                throw runtime_error("--q-index must be in [1,5]");
            endqi = startqi + 1;
        } else if (option == "--with-spf") {
            buildSPFIndices = true;
        } else if (option == "--spf-all-categories") {
            spfAllCategories = true;
        } else if (option == "--spf-legacy-filter") {
            spfLegacyCategoryFilter = true;
        } else if (option == "--spf-candidates" && argi + 1 < argc) {
            spfCandidateSize = stoi(argv[++argi]);
        } else if (option == "--spf-topk" && argi + 1 < argc) {
            spfTopk = stoi(argv[++argi]);
        } else if (option == "--spf-min-category-size" && argi + 1 < argc) {
            spfMinCategorySize = stoi(argv[++argi]);
        } else if (option == "--spf-seed" && argi + 1 < argc) {
            spfSeed = static_cast<unsigned int>(stoul(argv[++argi]));
        } else if (option == "--no-bounds") {
            enableBoundsPruning = false;
        } else if (option == "--no-spf") {
            enableSPFPruning = false;
        } else if (option == "--verify-exact") {
            verifyExact = true;
        } else if (option == "--hsm-candidates" && argi + 1 < argc) {
            hsmCandidateFile = argv[++argi];
        } else if (option == "--max-queries" && argi + 1 < argc) {
            maxQueries = stoi(argv[++argi]);
        } else {
            throw runtime_error("Unknown or incomplete option: " + option);
        }
    }
    cout << sfile << " " << reqfile << " " << alpha << endl;
    cout << "FSSP config: bounds=" << enableBoundsPruning
         << " spf=" << enableSPFPruning
         << " build_spf=" << buildSPFIndices
         << " verify_exact=" << verifyExact
         << " spf_scope=" << (spfAllCategories ? "all" : "matched")
         << " spf_candidates=" << spfCandidateSize
         << " spf_topk=" << spfTopk
         << " candidate_source=" << (hsmCandidateFile.empty() ? "embedding-threshold" : "hsm-file") << endl;

    FILE *fp_query, *fp_network;
    string prefix = string("../data/") + sfile + string("/");
    FILE *fp_requirement = fopen((prefix + reqfile).c_str(), "r");
    char buf[256];
    vector<string> R;
    while (fgets(buf, sizeof(buf), fp_requirement)) {
        string line(buf);
        if (!line.empty() && line[line.length()-1] == '\n') {
            line.erase(line.length()-1);
        }
        if (!line.empty() && line[line.length()-1] == '\r') {
            line.erase(line.length()-1);
        }
        if (!line.empty()) {
            R.push_back(line);
        }
    }
    fclose(fp_requirement);

    if (hsmCandidateFile.empty()) initTransformer();

    string graphfile = prefix + string("USA-road-d.") + sfile + (".gr");
    fp_network = fopen(graphfile.c_str(), "r");
    char ch, buffer[100];
    int u, v, intw;
    double w;
    char cat1;
    //read graph
    for (int i = 0; i < 4; i++)
        fgets(buffer, 90, fp_network);
    for (int i = 0; i < 4; i++)
        fgetc(fp_network);
    fscanf(fp_network, "%d%d", &N, &M);
    for (int i = 0; i < 3; i++)
        fgets(buffer, 90, fp_network);
    for (int i = 0; i < M; i++) {
        fscanf(fp_network, "%c%d%d%d", &ch, &u, &v, &intw);
        fgets(buffer, 90, fp_network);
        w = (double)intw / 10.0;
        u--;
        v--;
        //printf("%d %d %c%d %c\n", u, v, cat1, cat2, l);
        if (i % 2 == 0){
            G[u].push_back(ID(v, w));
            G[v].push_back(ID(u, w));
        }
    }

    double minx = DBL_MAX, miny = DBL_MAX, maxx = -DBL_MAX, maxy = -DBL_MAX;
    //coords
    string coordsfile = prefix + string("USA-road-d.") + sfile + (".co");
    FILE *fp_coords = fopen(coordsfile.c_str(), "r");
    for (int i = 0; i < 7; i++)
        fgets(buffer, 90, fp_coords);
    
    for (int i = 0; i < N; i++) {
        double x, y;
        fscanf(fp_coords, "%c%d%lf%lf", &ch, &u, &x, &y);
        fgets(buffer, 90, fp_coords);
        x /= 1000000.0;
        y /= 1000000.0;
        coords.push_back(DD(x, y));
        minx = min(x, minx), miny = min(y, miny), maxx = max(x, maxx), maxy = max(y, maxy);
    }
    //printf("%f %f %f %f\n", minx, maxx, miny, maxy);


    cout << "Building H2H Index for dataset: " << sfile << endl;
    // call h2h.cpp
    buildH2HIndex(sfile);
    cout << "H2H Index building completed!" << endl;

    std::chrono::high_resolution_clock::time_point t1, t2;
	std::chrono::duration<double> time_span;
	double runT;

    readPOINodeMapping(sfile);
    readCategoryID(sfile);
    if (hsmCandidateFile.empty()) readCategoryEmbeddings(sfile);

    t1=std::chrono::high_resolution_clock::now();
    buildCategoryRepresentatives(100);
    t2=std::chrono::high_resolution_clock::now();
    time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2-t1);
	runT= time_span.count();
    cout<<"BuildCategoryRepresentatives Time "<<runT<<endl;

    t1=std::chrono::high_resolution_clock::now();
    buildCategoryGrids(100);
    t2=std::chrono::high_resolution_clock::now();
    time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2-t1);
	runT= time_span.count();
    cout<<"BuildCategoryGrids Time "<<runT<<endl;

    //vector<string> R = {"beach", "desert shop", "Yoga Studio"};
    //vector<string> R = {"get some fast food", "buy a birthday gift", "fill up gas"};
    //vector<string> R = {"repair a tire", "buy some video games", "go to a flea market"};
    //vector<string> R = {"Restaurant", "Fast Food", "McDonald's", "Burger King", "Italian Food", "Shopping", "Gift Shop", "Electronics Store", "Bookstore", "Hospital", "Medical Center", "Gas Station", "Shell", "Pharmacy", "CVS"};

    vector<vector<double>> queryEmb;
    K = R.size();
    vector<vector<int> > similarCategories;
    if (hsmCandidateFile.empty()) {
        g_embeddingService->getEmbeddings(R, queryEmb);
        K = queryEmb.size();
        similarCategories.resize(K);
        for (int i = 0; i < K; ++i) {
            findSimilarCategories(queryEmb[i], alpha, similarCategories[i]);
            cout << "Requirement " << i + 1 << " matched " << similarCategories[i].size() << " categories" << endl;
        }
    } else {
        similarCategories = readHSMCandidateFile(hsmCandidateFile, K);
        for (int i = 0; i < K; ++i)
            cout << "Requirement " << i + 1 << " received " << similarCategories[i].size()
                 << " HSM category candidates" << endl;
    }

    if (buildSPFIndices && enableSPFPruning) {
        vector<int> spfCategories;
        if (spfAllCategories) {
            for (const auto& entry : category2Nodes) spfCategories.push_back(entry.first);
        } else {
            for (const auto& matched : similarCategories)
                spfCategories.insert(spfCategories.end(), matched.begin(), matched.end());
        }
        srand(spfSeed);
        t1=std::chrono::high_resolution_clock::now();
        buildSPFCategoryIndices(spfCategories, spfCandidateSize, spfTopk,
                                spfMinCategorySize, spfLegacyCategoryFilter);
        t2=std::chrono::high_resolution_clock::now();
        time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2-t1);
	runT= time_span.count();
        cout<<"BuildSPFCategoryIndices Time "<<runT<<endl;
    }

    vector<int> path;

    // 读取POI文件
    ifstream poi_file(prefix + "POInames_ID.txt");
    string line;
    
    while (getline(poi_file, line)) {
        istringstream iss(line);
        int id;
        string name;
        iss >> id; // 读取ID
        getline(iss, name); 
        
        if (!name.empty() && name[0] == ' ') {
            name = name.substr(1);
        }
        
        if (id >= poi_names.size()) {
            poi_names.resize(id + 1);
        }
        
        poi_names[id] = name;
    }
    
    poi_file.close();

    /*for (int qi = 0; qi < 1; qi++){//test a queryset
        vector<II> queryset;
        vector<double> ans;
        string s3 = string("../data/") + sfile + string("/") + string("q") + to_string(qi + 1);
        fp_query = fopen(s3.c_str(), "r");
        int qs, qt;
        while (~fscanf(fp_query, "%d%d", &qs, &qt)){
            queryset.push_back(II(qs, qt));
        }
        fclose(fp_query);
        int qn = queryset.size();
        t1=std::chrono::high_resolution_clock::now();
        for (int i = 0; i < queryset.size(); i++){
            ans.push_back(FSSPQuery(queryset[i].first, queryset[i].second, queryEmb, similarCategories, path));
        }
        t2=std::chrono::high_resolution_clock::now();
        time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2-t1);
        runT= time_span.count();
    }*/
    string setres;
    for (int qi = startqi; qi < endqi; qi++){
        vector<II> queryset;
        vector<double> ans;
        string s3 = string("../data/") + sfile + string("/") + string("q") + to_string(qi + 1);
        fp_query = fopen(s3.c_str(), "r");
        if (!fp_query) throw runtime_error("Cannot open query workload: " + s3);
        int qs, qt;
        while (~fscanf(fp_query, "%d%d", &qs, &qt)){
            queryset.push_back(II(qs, qt));
        }
        fclose(fp_query);
        int qn = queryset.size();
        prunedspf = prunedwub = nh2hcalls = 0;
        t1=std::chrono::high_resolution_clock::now();
        int runQueries = maxQueries < 0 ? qn : min(qn, maxQueries);
        if (runQueries == 0) throw runtime_error("Query workload is empty: " + s3);
        for (int i = 0; i < runQueries; i++)
        {
            int source = queryset[i].first - 1, target = queryset[i].second - 1;
            double distance = FSSPQuery(source, target, queryEmb, similarCategories, path);
            if (verifyExact) {
                double exact = exactDPQuery(source, target, similarCategories);
                if (fabs(distance - exact) > 1e-6) {
                    throw runtime_error("FSSP verification mismatch: " + to_string(distance) + " vs " + to_string(exact));
                }
            }
            ans.push_back(distance);
        }
        t2=std::chrono::high_resolution_clock::now();
        time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2-t1);
        runT= time_span.count();
        cout << (string("q") + to_string(qi + 1)).c_str() << " Query Time (ms, n=" << runQueries << ")\n" << runT*1000 << endl;
        cout << "PruningStats bounds=" << prunedwub << " spf=" << prunedspf << " distance_calls=" << nh2hcalls << endl;
        setres += string("q") + to_string(qi + 1);
        setres += string(" Query Time (ms, n=") + to_string(runQueries) + string(") \n") + to_string(runT*1000) + string("\n");
    }

    FILE *fp_record = fopen("FSSPRecord.txt", "a");
    fprintf(fp_record, "%s\n", sfile.c_str());
    fprintf(fp_record, "%s\n", setres.c_str());
    fclose(fp_record);

    return 0;
}
