// Shared graph/index/query driver. User explicitly chose the existing DROMC.cpp
// after its candidate-A* / paper-identity mismatch was disclosed. It is reported
// as DROMC (existing implementation), never as a verified full-paper reproduction.
#include <random>
#include <filesystem>
#include <unistd.h>
#include <sys/wait.h>
#include <poll.h>
#include <signal.h>
#define h2hQuery measuredDistance
#define main archivedFSSPMain
#include "source/FSSP.cpp"
#undef main
#undef h2hQuery
double h2hQuery(int,int);
#include "baseline_core.h"
#define h2hQuery measuredDistance
#define main archivedDROMCMain
namespace legacy_dromc {
#include "source_original/DROMC.cpp"
}
#undef main
#undef h2hQuery
extern map<int,vector<vector<int>>> category2anc;
void dijkstra_with_ancestors(int,vector<int>&,vector<vector<int>>&);
static bool fixtureMode=false;
static vector<vector<double>> floyd;
double measuredDistance(int a,int b) {
    ++workCounters.h2h_calls;return fixtureMode?floyd[a][b]:h2hQuery(a,b);
}
using Clock=chrono::steady_clock;
static RoseRNII rose;
double elapsed(Clock::time_point t){return chrono::duration<double,milli>(Clock::now()-t).count();}
bool same(double a,double b){return isfinite(a)&&isfinite(b)&&abs(a-b)<=1e-6+1e-9*abs(b);}
void ints(ostream& out,const vector<int>& v){out<<'[';for(size_t i=0;i<v.size();++i){if(i)out<<',';out<<v[i];}out<<']';}
struct Query {string id;int s,t;vector<int> cats;};
vector<Query> queries(const string& file) {
    ifstream in(file);if(!in)throw runtime_error("missing query list");
    vector<Query> result;string line;
    while(getline(in,line)) {
        istringstream row(line);Query q;int k;row>>q.id>>q.s>>q.t>>k;
        if(!row||k<1||k>5||q.s<0||q.t<0||q.s>=N||q.t>=N)throw runtime_error("invalid query");
        q.cats.resize(k);for(int& c:q.cats)if(!(row>>c))throw runtime_error("missing category");
        string more;if(row>>more)throw runtime_error("extra query field");result.push_back(q);
    }
    return result;
}
void loadGraph(const string& dataset) {
    string prefix="../data/"+dataset+"/",line;ifstream f(prefix+"USA-road-d."+dataset+".gr");
    if(!f)throw runtime_error("missing graph");
    int arcs=0,pu=-1,pv=-1,pw=-1;
    while(getline(f,line)) {
        if(line.empty()||line[0]=='c')continue;
        istringstream row(line);char ch;row>>ch;
        if(ch=='p'){string kind;row>>kind>>N>>M;if(N>=MAX_V)throw runtime_error("graph too large");}
        if(ch=='a') {
            int u,v,w;row>>u>>v>>w;--u;--v;
            if(u<0||v<0||u>=N||v>=N||w<=0)throw runtime_error("invalid graph edge");
            if(arcs%2==0){pu=u;pv=v;pw=w;G[u].emplace_back(v,w/10.);G[v].emplace_back(u,w/10.);}
            else if(u!=pv||v!=pu||w!=pw)throw runtime_error("graph differs from paired-undirected semantics");
            ++arcs;
        }
    }
    if(arcs!=M||arcs%2)throw runtime_error("invalid arc count");
    coords.assign(N,{});vector<unsigned char> seen(N,0);ifstream co(prefix+"USA-road-d."+dataset+".co");
    while(getline(co,line))if(!line.empty()&&line[0]=='v') {
        istringstream row(line);char ch;int v;double x,y;row>>ch>>v>>x>>y;--v;
        if(v<0||v>=N||seen[v])throw runtime_error("invalid coordinates");
        coords[v]={x/1e6,y/1e6};seen[v]=1;
    }
    if(count(seen.begin(),seen.end(),1)!=N)throw runtime_error("missing coordinates");
    readPOINodeMapping(dataset);readCategoryID(dataset);
}
void checkMemberships(const string& path) {
    ifstream in(path);int c,n,v;size_t verified=0;
    if(!in)throw runtime_error("missing candidate manifest");
    while(in>>c>>n) {
        vector<int> expected;for(int i=0;i<n;++i){if(!(in>>v))throw runtime_error("bad membership file");expected.push_back(v);}
        if(category2Nodes.at(c)!=expected)throw runtime_error("category membership differs from input freeze");
        ++verified;
    }
    if(!verified)throw runtime_error("empty membership file");
}
double runMethod(const string& method,const Query& q,vector<int>& path) {
    if(method=="FSSP") {
        // Existing getUpperBound() reads global K, unlike FSSPQuery's local k.
        // The original main sets it per workload; this adapter must set it per query.
        K=int(q.cats.size());
        vector<vector<int>> cats;for(int c:q.cats)cats.push_back({c});vector<vector<double>> embeddings;
        enableBoundsPruning=enableSPFPruning=true;prunedwub=prunedspf=nh2hcalls=0;
        return FSSPQuery(q.s,q.t,embeddings,cats,path);
    }
    if(method=="DROMC") {
        vector<vector<int>> cats;for(int c:q.cats)cats.push_back({c});vector<vector<double>> embeddings;
        // Query-local tables must not retain entries from different category
        // sequences. Initial reset and waypoint normalization are timed.
        // Retained cache cleanup is outside the timer for all methods.
        legacy_dromc::heuristicTable.clear();legacy_dromc::poiToDestDist.clear();
        vector<int> served;
        const double answer=legacy_dromc::DROMCQuery(q.s,q.t,embeddings,cats,served);
        path.clear();path.push_back(q.s);path.insert(path.end(),served.begin(),served.end());path.push_back(q.t);
        return answer;
    }
    auto groups=extractCandidates(q.cats);
    if(method=="LayeredDijkstra")return layeredQuery(q.s,q.t,groups,path);
    if(method=="ROSE") {
        return rose.query(q.s,q.t,groups,path);
    }
    throw runtime_error("unknown method");
}
void verifyPath(const Query& q,const vector<int>& path,double distance) {
    if(path.size()!=q.cats.size()+2||path.front()!=q.s||path.back()!=q.t)throw runtime_error("invalid waypoint output");
    for(size_t i=0;i<q.cats.size();++i) {
        auto& group=category2Nodes.at(q.cats[i]);
        if(!binary_search(group.begin(),group.end(),path[i+1]))throw runtime_error("waypoint not in category");
    }
    double sum=0;for(size_t i=1;i<path.size();++i)sum+=measuredDistance(path[i-1],path[i]);
    if(!same(sum,distance))throw runtime_error("waypoint cost mismatch");
}
void selfTest() {
    fixtureMode=true;mt19937 rng(42);int comparisons=0,multipleRefinements=0;
    for(int test=0;test<420;++test) {
        int old=N;N=test<4?3:3+rng()%9;
        for(int i=0;i<max(old,N);++i)G[i].clear();
        coords.resize(N);planeCoords.resize(N);category2Nodes.clear();category2Grids.clear();category2anc.clear();
        floyd.assign(N,vector<double>(N,DBL_MAX));
        for(int u=0;u<N;++u){coords[u]=planeCoords[u]={double(u),0};floyd[u][u]=0;}
        auto edge=[&](int u,int v,double w){G[u].emplace_back(v,w);G[v].emplace_back(u,w);floyd[u][v]=floyd[v][u]=min(floyd[u][v],w);};
        for(int u=1;u<N;++u)edge(u-1,u,test<4?1.:.1*(1+rng()%40));
        if(test>=4)for(int u=0;u<N;++u)for(int v=u+2;v<N;++v)if(rng()%3==0)edge(u,v,.1*(1+rng()%60));
        for(int z=0;z<N;++z)for(int u=0;u<N;++u)for(int v=0;v<N;++v)
            if(floyd[u][z]<DBL_MAX && floyd[z][v]<DBL_MAX)floyd[u][v]=min(floyd[u][v],floyd[u][z]+floyd[z][v]);
        K=test<4?2:1+rng()%5;nofcats=K;categoryRepresentatives.assign(K+1,{});
        vector<vector<int>> groups(K);Query q{"fixture",test<4?0:int(rng()%N),test==2?0:test<4?2:int(rng()%N),{}};
        for(int i=0;i<K;++i) {
            if(test==0)groups[i]={1};
            else if(test==1)groups[i]={i};
            else if(test==2)groups[i]={0};
            else if(test==3)groups[i]={2};
            else {for(int v=0;v<N;++v)if(rng()%3==0)groups[i].push_back(v);if(groups[i].empty())groups[i].push_back(rng()%N);}
            category2Nodes[i+1]=groups[i];q.cats.push_back(i+1);
            categoryRepresentatives[i+1]={groups[i][rng()%groups[i].size()]};
            for(int v:groups[i])category2Grids[i+1].push_back(Grid{planeCoords[v].first,planeCoords[v].first,planeCoords[v].second,planeCoords[v].second,{v}});
            for(int j=0;j<min(2,int(groups[i].size()));++j)dijkstra_with_ancestors(groups[i][j],category2Nodes[i+1],category2anc[i+1]);
        }
        geometricCertificate=safe_geometry::calibrate(planeCoords,G,N);
        rose.build(2);double reference=stateOracle(q.s,q.t,groups);
        for(const string method:{"FSSP","ROSE","DROMC","LayeredDijkstra"}) {
            // Poison legacy main-program state so the adapter itself must bind
            // every request's sequence length; covers mixed k=1..5 requests.
            K=0;
            workCounters={};vector<int> path;double answer=runMethod(method,q,path);
            if(!same(answer,reference)) {cerr<<"fixture="<<test<<" method="<<method<<" answer="<<answer<<" reference="<<reference<<endl;throw runtime_error("small graph mismatch");}
            if(method=="ROSE"&&workCounters.refinements>1)++multipleRefinements;
            verifyPath(q,path,answer);++comparisons;
        }
    }
    if(!multipleRefinements)throw runtime_error("ROSE recurrent refinement was not exercised");
    cout<<"{\"passed\":true,\"graph_cases\":420,\"algorithm_comparisons\":"<<comparisons
        <<",\"rose_multi_iteration_cases\":"<<multipleRefinements<<",\"semantics\":\"same-visit and endpoints allowed\"}"<<endl;
}
void oracleRun(const string& dataset,const string& input,const string& out) {
    loadGraph(dataset);checkMemberships(filesystem::path(input).parent_path().string()+"/candidate_memberships.txt");
    if(filesystem::exists(out))throw runtime_error("refuse to overwrite oracle output");
    ofstream f(out);f<<setprecision(17);auto qs=queries(input);int done=0;
    for(const auto& q:qs) {
        auto groups=extractCandidates(q.cats);double reference=stateOracle(q.s,q.t,groups);
        vector<int> path;double other=layeredQuery(q.s,q.t,groups,path);
        if(!same(reference,other))throw runtime_error("state and layered oracle mismatch");
        // Check waypoint segment costs using graph Dijkstra, independently of H2H.
        double total=0;for(size_t i=1;i<path.size();++i)total+=multiSource({{path[i-1],0}},{path[i]}).distance[path[i]];
        if(!same(total,reference))throw runtime_error("oracle waypoint cost mismatch");
        double direct=multiSource({{q.s,0}},{q.t}).distance[q.t];
        f<<"{\"id\":"<<quoted(q.id)<<",\"distance\":"<<reference<<",\"direct_graph_distance\":"<<direct<<",\"waypoints\":";ints(f,path);f<<"}\n";f.flush();
        if(++done%10==0)cout<<"ORACLE "<<dataset<<' '<<done<<'/'<<qs.size()<<endl;
    }
    cout<<"ORACLE_COMPLETE "<<done<<endl;
}

// A child process contains each query. Fork/IPC are outside the reported timer;
// all methods get the same fresh query-state policy and external timeout.
string isolated(const string& method,const Query& q,int budgetMs=30000) {
    int pipefd[2];if(pipe(pipefd)!=0)throw runtime_error("pipe failed");
    cout.flush();cerr.flush();pid_t pid=fork();if(pid<0)throw runtime_error("fork failed");
    if(pid==0) {
        close(pipefd[0]);const char begin='B';if(write(pipefd[1],&begin,1)!=1)_exit(90);
        try {
            workCounters={};vector<int> path;auto start=Clock::now();
            if(method=="timeout_fixture") {usleep(200000);_exit(91);}
            double distance=runMethod(method,q,path);const double millis=elapsed(start);const auto counters=workCounters;
            ostringstream out;out<<setprecision(17)<<"{\"id\":"<<quoted(q.id)<<",\"algorithm\":"<<quoted(method)
                <<",\"status\":\"ok\",\"planner_ms\":"<<millis<<",\"distance\":"<<distance
                <<",\"h2h_calls\":"<<counters.h2h_calls<<",\"settled_vertices\":"<<counters.settled
                <<",\"scanned_edges\":"<<counters.edges<<",\"pair_evaluations\":"<<counters.pair_evaluations
                <<",\"refinements\":"<<counters.refinements<<",\"waypoints\":";ints(out,path);out<<"}\n";
            string body=out.str();if(write(pipefd[1],body.data(),body.size())!=ssize_t(body.size()))_exit(92);
            close(pipefd[1]);_exit(0);
        }catch(const exception& e) {
            string body="{\"status\":\"error\",\"message\":\"query exception\"}\n";
            write(pipefd[1],body.data(),body.size());cerr<<e.what()<<endl;_exit(93);
        }
    }
    close(pipefd[1]);char begin;
    if(read(pipefd[0],&begin,1)!=1||begin!='B') {kill(pid,SIGKILL);waitpid(pid,nullptr,0);close(pipefd[0]);throw runtime_error("child handshake failed");}
    auto start=Clock::now();string body;bool timeout=false;
    for(;;) {
        int remaining=budgetMs-int(elapsed(start));if(remaining<=0){timeout=true;break;}
        pollfd p{pipefd[0],POLLIN,0};int ready=poll(&p,1,remaining);
        if(ready<0){if(errno==EINTR)continue;throw runtime_error("poll failed");}
        if(ready==0){timeout=true;break;}
        char buffer[4096];ssize_t n=read(pipefd[0],buffer,sizeof(buffer));
        if(n<=0)break;body.append(buffer,size_t(n));if(body.find('\n')!=string::npos)break;
    }
    if(timeout)kill(pid,SIGKILL);
    int code;waitpid(pid,&code,0);close(pipefd[0]);
    if(timeout) {ostringstream out;out<<"{\"id\":"<<quoted(q.id)<<",\"algorithm\":"<<quoted(method)<<",\"status\":\"timeout\",\"planner_ms\":null,\"censor_ms\":"<<budgetMs<<"}\n";return out.str();}
    if(!WIFEXITED(code)||WEXITSTATUS(code)!=0)throw runtime_error("child failed: "+body);
    return body;
}
void worker(const string& dataset,const string& input) {
    loadGraph(dataset);checkMemberships(filesystem::path(input).parent_path().string()+"/candidate_memberships.txt");
    buildH2HIndex(dataset);srand(42);buildCategoryRepresentatives(100);buildCategoryGrids(100);
    if(!geometricCertificate.ready||geometricCertificate.verification_violations)throw runtime_error("unsafe geometry");
    vector<int> cs;for(const auto& [c,nodes]:category2Nodes)cs.push_back(c);
    srand(42);buildSPFCategoryIndices(cs,4,2,1,false);rose.build(50);
    map<string,Query> manifest;for(auto q:queries(input))manifest[q.id]=q;
    cout<<"READY "<<dataset<<' '<<manifest.size()<<endl;
    string method,id;
    while(cin>>method>>id) {
        if(method=="QUIT")break;
        if(!manifest.count(id))throw runtime_error("unknown query ID");
        if(method=="VERIFY") {
            double d;int n;cin>>d>>n;if(n<2||n>7)throw runtime_error("invalid verification path");
            vector<int> path(n);for(int& v:path)cin>>v;
            verifyPath(manifest.at(id),path,d);cout<<"VERIFY_OK "<<id<<endl;continue;
        }
        cout<<isolated(method,manifest.at(id))<<flush;
    }
}
int main(int argc,char** argv) {
    try {
        cout<<setprecision(17);
        if(argc==2 && string(argv[1])=="selftest") {selfTest();return 0;}
        if(argc==2 && string(argv[1])=="timeout-test") {
            const string r=isolated("timeout_fixture",Query{"timeout_fixture",0,0,{}},20);
            if(r.find("\"status\":\"timeout\"")==string::npos)throw runtime_error("timeout not censored");cout<<r;return 0;
        }
        if(argc==5 && string(argv[1])=="oracle") {oracleRun(argv[2],argv[3],argv[4]);return 0;}
        if(argc==4 && string(argv[1])=="worker") {worker(argv[2],argv[3]);return 0;}
        throw runtime_error("usage: selftest | timeout-test | oracle DATASET QUERIES OUTPUT | worker DATASET QUERIES");
    }catch(const exception& e){cerr<<"FAILED "<<e.what()<<endl;return 2;}
}
