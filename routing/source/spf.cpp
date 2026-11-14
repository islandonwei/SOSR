#include "spf.h"
vector<ID> G[MAX_V];
map<int, vector<int>> category2Nodes;
map<int, vector<vector<int>>> category2anc;

void dijkstra_with_ancestors(int start, vector<int>& S_set, vector<vector<int>> &anc) {
    vector<bool> in_S(N, false);
    for(int s : S_set) {
        in_S[s] = true;
    }
    vector<int> tmp(N, -1);
    anc.push_back(tmp);

    vector<double> dist(N, DBL_MAX);
    vector<bool> visited(N, false);
    vector<int> current_s_ancestor(N, -1);  // 当前路径上最近的S中的点

    priority_queue<pair<double, int>, vector<pair<double, int>>, greater<pair<double, int>>> pq;
    
    dist[start] = 0.0;
    pq.push({0.0, start});
    
    // 如果起点在S中
    if(in_S[start]) {
        current_s_ancestor[start] = start;
        anc[anc.size()-1][start] = -1;  // 起点没有祖先
    }
    
    while (!pq.empty()) {
        auto [d, u] = pq.top();
        pq.pop();
        
        if (visited[u]) continue;
        if(dist[u] < d) continue;
        
        visited[u] = true;
        
        for (auto& edge : G[u]) {
            int v = edge.first;
            double w = edge.second;
            
            if (!visited[v] && dist[u] + w < dist[v]) {
                dist[v] = dist[u] + w;
                
                // 更新祖先信息
                if(in_S[v]) {
                    anc[anc.size()-1][v] = current_s_ancestor[u];  // v的祖先是路径上前一个S中的点
                    current_s_ancestor[v] = v;       // v成为新的当前S祖先
                } else {
                    anc[anc.size()-1][v] = current_s_ancestor[u];
                    current_s_ancestor[v] = current_s_ancestor[u];  // 继承父节点的S祖先
                }
                
                pq.push({dist[v], v});
            }
        }
    }    
}

double calculatePruningScore(const vector<int>& anc_indices, const vector<int>& Sset, const vector<int>& sample_points, vector<vector<int>> &anc) {
    double total_pruned = 0;
    
    for(int v : sample_points){
        set<int> prunedv;
        for(int anc_idx : anc_indices) {
            int start_point;
            if (anc[anc_idx][v] != -1)
                start_point = anc[anc_idx][v];
            else
                continue;
            
            int current = anc[anc_idx][start_point];
            while(current != -1) {
                prunedv.insert(current);
                current = anc[anc_idx][current];
            }
        }
        total_pruned += prunedv.size();
    }
    
    return total_pruned / sample_points.size();
}

double calculateSingleStartScore(int start_idx, const vector<int>& query_points, const vector<int>& Sset, vector<vector<int>> &anc) {
    double total_pruned = 0;
    
    for(int v : query_points) {
        set<int> prunedv;
        
        int start_point;
        if (anc[start_idx][v] != -1)
            start_point = anc[start_idx][v];
        else
            continue;
        
        int current = anc[start_idx][start_point];
        while(current != -1) {
            prunedv.insert(current);
            current = anc[start_idx][current];
        }
        total_pruned += prunedv.size();
    }
    
    return total_pruned / query_points.size();
}

double calculateIncrementalScore(int candidate_start_idx, 
                                vector<set<int>>& already_pruned_per_query,
                                const vector<int>& query_points, 
                                const vector<int>& Sset, vector<vector<int>> &anc) {
    double incremental_pruned = 0;
    
    for(int i = 0; i < query_points.size(); i++) {
        int v = query_points[i];
        set<int>& already_pruned = already_pruned_per_query[i];
        
        // 计算候选start能过滤的新点
        int start_point;
        if (anc[candidate_start_idx][v] != -1)
            start_point = anc[candidate_start_idx][v];
        else
            continue;
        
        int current = anc[candidate_start_idx][start_point];
        while(current != -1) {
            if(already_pruned.find(current) == already_pruned.end()) {
                incremental_pruned++;
            }
            current = anc[candidate_start_idx][current];
        }
    }
    
    return incremental_pruned / query_points.size();
}

// 新增函数：更新已过滤集合
void updatePrunedSets(int selected_start_idx, 
                      vector<set<int>>& already_pruned_per_query,
                      const vector<int>& query_points, vector<vector<int>> &anc) {
    for(int i = 0; i < query_points.size(); i++) {
        int v = query_points[i];
        set<int>& already_pruned = already_pruned_per_query[i];
        
        int start_point;
        if (anc[selected_start_idx][v] != -1)
            start_point = anc[selected_start_idx][v];
        else
            continue;
        
        int current = anc[selected_start_idx][start_point];
        while(current != -1) {
            already_pruned.insert(current);
            current = anc[selected_start_idx][current];
        }
    }
}

void greedySetCover(const unordered_map<int,int>& startSet, const vector<int>& Sset, 
                   int max_starts, int sample_size, vector<vector<int>>& anc) {
    // 生成随机查询点
    vector<int> query_points;
    for(int i = 0; i < sample_size && i < coords.size(); i++) {
        query_points.push_back(rand() % coords.size());
    }
    
    vector<int> candidates;
    for(auto& p : startSet) {
        candidates.push_back(p.second); // anc数组索引
    }
    
    vector<int> selected_starts;
    vector<set<int>> already_pruned_per_query(query_points.size());
    
    // 第一轮：选择单独效果最好的start
    int best_start = -1;
    double best_score = 0;
    int best_idx = -1;
    
    for(int i = 0; i < candidates.size(); i++) {
        int start_idx = candidates[i];
        double score = calculateSingleStartScore(start_idx, query_points, Sset, anc);
        if(score > best_score) {
            best_score = score;
            best_start = start_idx;
            best_idx = i;
        }
    }
    
    if(best_start != -1) {
        selected_starts.push_back(best_start);
        updatePrunedSets(best_start, already_pruned_per_query, query_points, anc);
        
        swap(candidates[best_idx], candidates.back());
        candidates.pop_back();
    }
    
    // 后续轮次：贪心选择增量效果最好的start
    for(int round = 1; round < max_starts && !candidates.empty(); round++) {
        int best_incremental_start = -1;
        double best_incremental_score = 0;
        best_idx = -1;
        
        for(int i = 0; i < candidates.size(); i++) {
            int candidate_idx = candidates[i];
            double incremental_score = calculateIncrementalScore(
                candidate_idx, already_pruned_per_query, query_points, Sset, anc
            );
            
            if(incremental_score > best_incremental_score) {
                best_incremental_score = incremental_score;
                best_incremental_start = candidate_idx;
                best_idx = i;
            }
        }
        
        if(best_incremental_start != -1 && best_incremental_score > 0.01) {
            selected_starts.push_back(best_incremental_start);
            updatePrunedSets(best_incremental_start, already_pruned_per_query, query_points, anc);
            
            swap(candidates[best_idx], candidates.back());
            candidates.pop_back();
        } else {
            break;
        }
    }
    
    // Preserve exactly the selected arrays.  The earlier swap-and-pop code
    // invalidated indices stored in selected_starts after the first swap,
    // producing an arbitrary subset rather than the greedy solution.
    vector<vector<int>> selected;
    selected.reserve(selected_starts.size());
    for (int index : selected_starts) selected.push_back(std::move(anc[index]));
    anc.swap(selected);
    
    //cout << "Kept " << selected_starts.size() << " out of " << candidates.size() + selected_starts.size() << " starts" << endl;
}

void buildSPFCatIndex(int catid, int candidate_size, int topk, vector<vector<int>>& anc) {
    unordered_map<int,int> startset;
    int ncat2nodes = category2Nodes[catid].size();
    
    for (int i = 0; i < candidate_size; i++) {
        int start = category2Nodes[catid][rand() % ncat2nodes];
        if(startset.count(start))
            continue;
        startset[start] = anc.size();
        dijkstra_with_ancestors(start, category2Nodes[catid], anc);
    }
    greedySetCover(startset, category2Nodes[catid], topk, 100, anc);
    cout << "Category " << catid << ": final " << anc.size() << " starts and Space " << anc.size() * N * 4 / (1024.0 * 1024.0 * 1024.0) << " GB" << endl;
    /*
    vector<int> sample_points;
    FILE *fp_random = fopen("RandQPs4Scores.txt", "r");
    if (fp_random) {
        for (int i = 0; i < 1000; i++){
            int a;
            if (fscanf(fp_random, "%d", &a) == 1) {
                sample_points.push_back(a);
            }
        }
        fclose(fp_random);
        
        vector<int> all_indices;
        for (int i = 0; i < anc.size(); i++) {
            all_indices.push_back(i);
        }
        
        cout << "Final pruning score: " << calculatePruningScore(all_indices, category2Nodes[catid], sample_points, anc) << endl;
    }    
    cout << "Space " << anc.size() * N * 4 / (1024.0 * 1024.0 * 1024.0) << " GB" << endl;*/
}

void calculateCategory2AncMemory() {
    size_t total_bytes = 0;
    
    for (auto& [catid, anc_arrays] : category2anc) {
        // map entry overhead (key + pointer to value)
        total_bytes += sizeof(int) + sizeof(vector<vector<int>>);
        
        // outer vector overhead
        total_bytes += sizeof(vector<int>) * anc_arrays.capacity();
        
        // inner vectors (anc arrays)
        for (auto& anc_array : anc_arrays) {
            total_bytes += sizeof(int) * anc_array.capacity();
        }
    }
    
    // map overhead
    total_bytes += sizeof(category2anc);
    double gb = total_bytes / (1024.0 * 1024.0 * 1024.0);
    cout << "Total memory: " << gb << " GB" << endl;
    cout << "=================================" << endl;
}

bool notperformSPF(int n, int cn){
    if ((n < 1000000) && ((cn > 400 && cn < 800) || (cn > 5000)))
        return 0;
    if((n > 1000000) && ((cn > 500 && cn < 600) || (cn > 10000)))
        return 0;
    return 1;
}

void buildSPFCategoryIndices(const vector<int>& categories, int candidateSize,
                             int topk, int minCategorySize,
                             bool useLegacyCategoryFilter) {
    if (candidateSize <= 0 || topk <= 0)
        throw runtime_error("SPF candidate size and top-k must be positive");
    category2anc.clear();
    set<int> uniqueCategories(categories.begin(), categories.end());
    cout << "Building SPF indices for " << uniqueCategories.size()
         << " requested categories (candidates=" << candidateSize
         << ", topk=" << topk << ", min_category_size=" << minCategorySize
         << ", legacy_filter=" << useLegacyCategoryFilter << ")..." << endl;
    for (int catid : uniqueCategories) {
        auto nodeIt = category2Nodes.find(catid);
        if (nodeIt == category2Nodes.end() || nodeIt->second.empty()) continue;
        const vector<int>& nodeList = nodeIt->second;
        if (static_cast<int>(nodeList.size()) < minCategorySize ||
            (useLegacyCategoryFilter && notperformSPF(N, nodeList.size()))) continue;
        cout << "Processing category " << catid << " with " << nodeList.size() << " nodes" << endl;
        vector<vector<int>> anc;
        buildSPFCatIndex(catid, candidateSize, topk, anc);
        if (!anc.empty()) category2anc[catid] = std::move(anc);
    }
    calculateCategory2AncMemory();
    cout << "SPF indices built for " << category2anc.size() << " categories" << endl;
}

static bool flag[MAX_V];
vector<int> applySPFPruning(int u, const vector<int>& Vr, vector<int>& similarCategories) {
    // 用unordered_map管理Vr中的点，方便删除
    unordered_map<int, int> vr_map;
    for (int v : Vr) {
        flag[v] = 1;
    }
    
    // 遍历similarCategories[round]里的每一个catid
    for (int catid : similarCategories) {
        // 检查该category是否有SPF索引
        if (category2anc.find(catid) == category2anc.end()) continue;
        
        // 遍历category2anc[catid]中的每一个anc数组
        for (auto& anc_array : category2anc[catid]) {
            // 以u开始找祖先链
            int start_point;
            if (anc_array[u] != -1)
                start_point = anc_array[u];  // u的第一个祖先（保留）
            else
                continue;
            
            // 从第一个祖先的祖先开始过滤（后续祖先过滤）
            int current = anc_array[start_point];
            while (current != -1) {
                flag[current] = 0;
                current = anc_array[current];
            }
        }
    }
    
    // 构建返回的vector，遍历剩余的unordered_map
    vector<int> filtered_Vr;

    for (int v : Vr) {
        if(flag[v])
            filtered_Vr.push_back(v);
        flag[v] = 0;
    }
    
    return filtered_Vr;
}
