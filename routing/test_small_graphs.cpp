// Actual copied FSSP DP/pruning with a Floyd distance oracle, compared against
// independent graph-state Dijkstra. No real workload or historical binary writes.
#define h2hQuery fixtureDistance
#define main archivedFSSPMain
#include "source/FSSP.cpp"
#undef main
#undef h2hQuery
#include <random>
#include <iomanip>
extern map<int,vector<vector<int>>> category2anc;
void dijkstra_with_ancestors(int,vector<int>&,vector<vector<int>>&);
static vector<vector<double>> fixtureDistances;
double fixtureDistance(int a,int b){return fixtureDistances[a][b];}

double stateOracle(int s,int t,const vector<vector<int>>& groups){
 const int k=groups.size();vector<vector<double>> d(N,vector<double>(k+1,DBL_MAX));
 using State=tuple<double,int,int>;priority_queue<State,vector<State>,greater<State>> q;
 d[s][0]=0;q.emplace(0,s,0);
 while(!q.empty()){
  auto [cost,u,i]=q.top();q.pop();if(cost!=d[u][i])continue;
  if(u==t&&i==k)return cost;
  if(i<k&&find(groups[i].begin(),groups[i].end(),u)!=groups[i].end()&&cost<d[u][i+1]){
   d[u][i+1]=cost;q.emplace(cost,u,i+1);
  }
  for(auto [v,w]:G[u])if(cost+w<d[v][i]){d[v][i]=cost+w;q.emplace(d[v][i],v,i);}
 }
 return DBL_MAX;
}

double layeredOracle(int s,int t,const vector<vector<int>>& groups){
 vector<double> prior(N,DBL_MAX);prior[s]=0;
 for(const auto& group:groups){
  vector<double> d=prior;priority_queue<DI,vector<DI>,greater<DI>> q;
  for(int v=0;v<N;++v)if(d[v]<DBL_MAX)q.emplace(d[v],v);
  while(!q.empty()){
   auto [cost,u]=q.top();q.pop();if(cost!=d[u])continue;
   for(auto [v,w]:G[u])if(cost+w<d[v]){d[v]=cost+w;q.emplace(d[v],v);}
  }
  fill(prior.begin(),prior.end(),DBL_MAX);for(int v:group)prior[v]=d[v];
 }
 vector<double> d=prior;priority_queue<DI,vector<DI>,greater<DI>> q;
 for(int v=0;v<N;++v)if(d[v]<DBL_MAX)q.emplace(d[v],v);
 while(!q.empty()){
  auto [cost,u]=q.top();q.pop();if(cost!=d[u])continue;if(u==t)return cost;
  for(auto [v,w]:G[u])if(cost+w<d[v]){d[v]=cost+w;q.emplace(d[v],v);}
 }
 return DBL_MAX;
}

int main(){
 mt19937 random(42);int cases=0,comparisons=0,infeasible=0;
 cout<<setprecision(17);
 for(int fixture=0;fixture<840;++fixture){
  int oldN=N;N=fixture<4?3:3+random()%9;
  for(int i=0;i<max(oldN,N);++i)G[i].clear();
  coords.resize(N);planeCoords.resize(N);category2Nodes.clear();category2Grids.clear();category2anc.clear();
  fixtureDistances.assign(N,vector<double>(N,DBL_MAX));
  for(int i=0;i<N;++i){
   coords[i]=planeCoords[i]=fixture<420?DD{double(i),0.0}:DD{1e7+double(int(random()%2001)-1000)/3.,3e6+double(int(random()%2001)-1000)/7.};
   fixtureDistances[i][i]=0;
  }
  auto edge=[&](int u,int v,double w){G[u].emplace_back(v,w);G[v].emplace_back(u,w);fixtureDistances[u][v]=fixtureDistances[v][u]=min(fixtureDistances[u][v],w);};
  for(int i=1;i<N;++i)edge(i-1,i,fixture<4?1:1+random()%4);
  if(fixture>=4)for(int u=0;u<N;++u)for(int v=u+2;v<N;++v)if(random()%3==0)edge(u,v,double(v-u)+random()%5);
  for(int z=0;z<N;++z)for(int u=0;u<N;++u)for(int v=0;v<N;++v)
   if(fixtureDistances[u][z]<DBL_MAX&&fixtureDistances[z][v]<DBL_MAX)fixtureDistances[u][v]=min(fixtureDistances[u][v],fixtureDistances[u][z]+fixtureDistances[z][v]);
  K=fixture<4?2:1+random()%5;nofcats=K;categoryRepresentatives.assign(K+1,{});
  vector<vector<int>> groups(K),similar(K);vector<vector<double>> embeddings;
  for(int i=0;i<K;++i){
   if(fixture==0)groups[i]={1};
   else if(fixture==1)groups[i]={i};
   else if(fixture==2)groups[i]={0};
   else if(fixture==3)groups[i]=i==0?vector<int>{1}:vector<int>{};
   else{for(int v=0;v<N;++v)if(random()%3==0)groups[i].push_back(v);if(groups[i].empty())groups[i].push_back(random()%N);}
   category2Nodes[i+1]=groups[i];similar[i]={i+1};
   if(!groups[i].empty())categoryRepresentatives[i+1]={groups[i][random()%groups[i].size()]};
   for(int v:groups[i])category2Grids[i+1].push_back(Grid{planeCoords[v].first,planeCoords[v].first,planeCoords[v].second,planeCoords[v].second,{v}});
   for(int j=0;j<min(2,int(groups[i].size()));++j)dijkstra_with_ancestors(groups[i][j],category2Nodes[i+1],category2anc[i+1]);
  }
  geometricCertificate=safe_geometry::calibrate(planeCoords,G,N);
  for(int u=0;u<N;++u)for(int v=0;v<N;++v){
   double lower=safe_geometry::routeLower(geometricCertificate,safe_geometry::rectangleDistanceLower(planeCoords[u].first,planeCoords[u].second,planeCoords[v].first,planeCoords[v].first,planeCoords[v].second,planeCoords[v].second),0);
   if(lower>fixtureDistances[u][v]+1e-12){cerr<<"GEOMETRIC_PAIR_BOUND_FAILURE"<<endl;return 8;}
  }
  int s=fixture<4?0:random()%N,t=fixture==2?0:fixture<4?2:random()%N;
  double expected=stateOracle(s,t,groups),layered=layeredOracle(s,t,groups);
  if(expected!=layered){cerr<<"LAYERED_MISMATCH "<<fixture<<endl;return 2;}
  for(int variant=0;variant<4;++variant){
   enableBoundsPruning=!(variant&1);enableSPFPruning=!(variant&2);vector<int> path;
   try{
    double actual=FSSPQuery(s,t,embeddings,similar,path);
    if(expected==DBL_MAX||abs(actual-expected)>1e-8){cerr<<"FSSP_MISMATCH fixture="<<fixture<<" variant="<<variant<<" expected="<<expected<<" actual="<<actual<<endl;return 3;}
    if(path.size()!=size_t(K+2)||path.front()!=s||path.back()!=t){cerr<<"PATH_FORMAT_FAILURE"<<endl;return 4;}
    double total=0;for(int i=0;i<K;++i){if(find(groups[i].begin(),groups[i].end(),path[i+1])==groups[i].end())return 5;}
    for(size_t i=1;i<path.size();++i)total+=fixtureDistance(path[i-1],path[i]);
    if(abs(total-actual)>1e-8)return 6;
   }catch(const exception& e){if(expected!=DBL_MAX){cerr<<"UNEXPECTED_EXCEPTION "<<fixture<<" "<<e.what()<<endl;return 7;}++infeasible;}
   ++comparisons;
  }
  ++cases;
 }
 cout<<"{\"passed\":true,\"graph_query_cases\":"<<cases<<",\"FSSP_variant_comparisons\":"<<comparisons<<",\"infeasible_cases_handled\":"<<infeasible<<",\"seed\":42,\"distance_oracle\":\"Floyd for copied FSSP; independent state and layered Dijkstra on graph edges\",\"semantics\":\"nondecreasing visits including source and target\",\"real_H2H_and_formal_workloads_verified\":false}"<<endl;
}
