#pragma once
// ROSE: Zhu et al., DSE 2022, section 5.3. Static, unweighted-rating,
// fixed-category specialization. This is a documented reimplementation,
// not the old ROSE.cpp exact-pair-DP substitution or official authors' code.
// All methods use the same nonnegative graph and weakly ordered visit semantics.

struct WorkCounters {
    uint64_t h2h_calls=0, settled=0, edges=0, pair_evaluations=0, refinements=0;
};
static WorkCounters workCounters;
double measuredDistance(int,int);

vector<vector<int>> extractCandidates(const vector<int>& cats) {
    vector<vector<int>> groups;
    for(int c:cats) {
        const auto it=category2Nodes.find(c);
        if(it==category2Nodes.end() || it->second.empty()) throw runtime_error("empty category");
        groups.push_back(it->second);
    }
    return groups;
}

struct SearchWave { vector<double> distance; vector<int> root; };
SearchWave multiSource(const vector<pair<int,double>>& seeds,const vector<int>& targets) {
    SearchWave result{vector<double>(N,DBL_MAX),vector<int>(N,-1)};
    vector<unsigned char> wanted(N,0);
    size_t remaining=0;
    for(int v:targets)if(!wanted[v]) {wanted[v]=1;++remaining;}
    priority_queue<DI,vector<DI>,greater<DI>> queue;
    for(auto [v,initial]:seeds)if(initial<result.distance[v]) {
        result.distance[v]=initial;result.root[v]=v;queue.emplace(initial,v);
    }
    while(!queue.empty() && remaining) {
        auto [d,u]=queue.top();queue.pop();
        if(d!=result.distance[u])continue;
        ++workCounters.settled;
        if(wanted[u]) {wanted[u]=0;if(--remaining==0)break;}
        for(auto [v,w]:G[u]) {
            ++workCounters.edges;
            if(d+w<result.distance[v]) {
                result.distance[v]=d+w;result.root[v]=result.root[u];queue.emplace(d+w,v);
            }
        }
    }
    if(remaining)throw runtime_error("unreachable target");
    return result;
}

double layeredQuery(int s,int t,const vector<vector<int>>& groups,vector<int>& path) {
    vector<pair<int,double>> seeds{{s,0}};
    vector<vector<int>> parent;
    for(const auto& group:groups) {
        auto wave=multiSource(seeds,group);
        seeds.clear();vector<int> roots;
        for(int v:group) {seeds.emplace_back(v,wave.distance[v]);roots.push_back(wave.root[v]);}
        parent.push_back(std::move(roots));
    }
    auto last=multiSource(seeds,{t});
    int node=last.root[t];path.assign(groups.size()+2,-1);path.back()=t;
    for(int i=int(groups.size())-1;i>=0;--i) {
        path[i+1]=node;
        auto pos=lower_bound(groups[i].begin(),groups[i].end(),node);
        if(pos==groups[i].end()||*pos!=node)throw runtime_error("missing layered parent");
        node=parent[i][pos-groups[i].begin()];
    }
    path[0]=node;if(node!=s)throw runtime_error("invalid layered source");
    return last.distance[t];
}

// Independently implemented graph-state oracle, with explicit zero-cost
// transitions between requirement levels. Not timed as an algorithm baseline.
double stateOracle(int s,int t,const vector<vector<int>>& groups) {
    const size_t levels=groups.size()+1;
    vector<double> distances(size_t(N)*levels,DBL_MAX);
    vector<vector<unsigned char>> member(groups.size(),vector<unsigned char>(N,0));
    for(size_t i=0;i<groups.size();++i)for(int v:groups[i])member[i][v]=1;
    using Item=tuple<double,int,int>;
    priority_queue<Item,vector<Item>,greater<Item>> queue;
    distances[size_t(s)*levels]=0;queue.emplace(0,s,0);
    while(!queue.empty()) {
        auto [d,u,i]=queue.top();queue.pop();
        if(d!=distances[size_t(u)*levels+i])continue;
        if(u==t && size_t(i)==groups.size())return d;
        if(size_t(i)<groups.size() && member[i][u] && d<distances[size_t(u)*levels+i+1]) {
            distances[size_t(u)*levels+i+1]=d;queue.emplace(d,u,i+1);
        }
        for(auto [v,w]:G[u])if(d+w<distances[size_t(v)*levels+i]) {
            distances[size_t(v)*levels+i]=d+w;queue.emplace(d+w,v,i);
        }
    }
    throw runtime_error("infeasible oracle input");
}

class RoseRNII {
    vector<int> landmarks;
    vector<vector<double>> rows;
    unordered_map<uint64_t,double> exact;
    static uint64_t pairKey(int a,int b) {
        if(a>b)swap(a,b);return (uint64_t(uint32_t(a))<<32)|uint32_t(b);
    }
    const vector<double>& row(int v) {
        auto& r=rows[v];
        if(r.empty()) {r.reserve(landmarks.size());for(int l:landmarks)r.push_back(measuredDistance(v,l));}
        return r;
    }
    double estimate(int a,int b) {
        ++workCounters.pair_evaluations;
        if(a==b)return 0;
        auto found=exact.find(pairKey(a,b));if(found!=exact.end())return found->second;
        const auto& x=row(a);const auto& y=row(b);double lower=0;
        for(size_t i=0;i<x.size();++i)lower=max(lower,abs(x[i]-y[i]));
        // Downward slack guards binary64 cancellation in landmark differences.
        return max(0.,lower-1e-7);
    }
    double exactDistance(int a,int b) {
        if(a==b)return 0;
        const auto key=pairKey(a,b);auto found=exact.find(key);
        if(found!=exact.end())return found->second;
        return exact[key]=measuredDistance(a,b);
    }
    double guide(int s,int t,const vector<vector<int>>& groups,double upper,vector<int>& path) {
        vector<int> prev{s};vector<double> costs{0};vector<vector<int>> parents;
        for(const auto& group:groups) {
            vector<double> next(group.size(),DBL_MAX);vector<int> par(group.size(),-1);
            for(size_t j=0;j<group.size();++j)for(size_t a=0;a<prev.size();++a) {
                if(costs[a]>=upper)continue;
                const double candidate=costs[a]+estimate(prev[a],group[j]);
                if(candidate<upper && candidate<next[j]) {next[j]=candidate;par[j]=int(a);}
            }
            parents.push_back(std::move(par));prev=group;costs=std::move(next);
        }
        double lower=DBL_MAX;int best=-1;
        for(size_t a=0;a<prev.size();++a)if(costs[a]<upper) {
            double candidate=costs[a]+estimate(prev[a],t);
            if(candidate<lower) {lower=candidate;best=int(a);}
        }
        path.clear();if(best<0)return DBL_MAX;
        path.assign(groups.size()+2,-1);path.front()=s;path.back()=t;
        for(int i=int(groups.size())-1;i>=0;--i) {
            if(best<0)throw runtime_error("ROSE missing predecessor");
            path[i+1]=groups[i][best];best=parents[i][best];
        }
        return lower;
    }
public:
    void build(int budget=50) {
        rows.assign(N,{});landmarks.clear();exact.clear();
        set<int> unique;for(const auto& [c,vs]:category2Nodes)unique.insert(vs.begin(),vs.end());
        vector<int> pois(unique.begin(),unique.end());if(pois.empty())throw runtime_error("no POIs");
        // Same farthest-first landmark rule as the existing reproduction,
        // but incrementally maintain minimum distances instead of recomputing.
        mt19937 generator(42);size_t selected=generator()%pois.size();
        vector<double> closest(pois.size(),DBL_MAX);vector<unsigned char> used(pois.size(),0);
        for(int i=0;i<min(budget,int(pois.size()));++i) {
            const int landmark=pois[selected];landmarks.push_back(landmark);used[selected]=1;
            for(size_t j=0;j<pois.size();++j) {
                double d=measuredDistance(pois[j],landmark);rows[pois[j]].push_back(d);closest[j]=min(closest[j],d);
            }
            double farthest=-1;
            for(size_t j=0;j<pois.size();++j)if(!used[j] && closest[j]>farthest) {farthest=closest[j];selected=j;}
        }
    }
    double query(int s,int t,const vector<vector<int>>& groups,vector<int>& path) {
        exact.clear();
        // Query-specific source/target rows are included in online timing.
        row(s);row(t);
        vector<int> incumbent{s};double upper=0;int u=s;
        for(const auto& group:groups) {
            int best=-1;double cost=DBL_MAX;
            for(int v:group) {double d=exactDistance(u,v);if(d<cost) {cost=d;best=v;}}
            if(best<0)throw runtime_error("ROSE empty group");
            upper+=cost;u=best;incumbent.push_back(u);
        }
        upper+=exactDistance(u,t);incumbent.push_back(t);
        for(;;) {
            ++workCounters.refinements;vector<int> guiding;
            const double lower=guide(s,t,groups,upper,guiding);
            if(guiding.empty() || lower>=upper-1e-7) {path=incumbent;return upper;}
            double actual=0;
            for(size_t i=1;i<guiding.size();++i)actual+=exactDistance(guiding[i-1],guiding[i]);
            if(actual<upper) {upper=actual;incumbent=guiding;}
            if(abs(actual-lower)<=1e-7) {path=incumbent;return upper;}
            // No arbitrary iteration cap: timeout is an external censored result.
        }
    }
    size_t landmarkCount() const {return landmarks.size();}
    void releaseQueryState() { exact.clear(); }
};
