#include "source/safe_geometry.h"
#include <iostream>
#include <random>
#include <string>
using XY = std::vector<std::pair<double,double>>;
using Graph = std::vector<std::vector<std::pair<int,double>>>;
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
template<class F> void mustReject(F fn) {
    bool rejected = false;
    try { fn(); } catch (const std::runtime_error&) { rejected = true; }
    require(rejected, "Invalid input was not rejected");
}
int main() {
    using namespace safe_geometry;
    std::mt19937_64 random(20261019);
    std::uniform_real_distribution<double> coord(-1e7, 1e7), offset(-1000, 1000);
    std::size_t norms = 0, rectangles = 0;
    for (int i = 0; i < 20000; ++i) {
        const double ax = coord(random), ay = coord(random);
        const double bx = ax + offset(random), by = ay + offset(random);
        const long double dx = static_cast<long double>(ax) - bx;
        const long double dy = static_cast<long double>(ay) - by;
        const long double reference = std::hypot(dx, dy);
        require(static_cast<long double>(normUpper(ax, ay, bx, by)) >= reference, "Edge norm upper endpoint underestimates");
        ++norms;
        const double minX = std::min(ax,bx), maxX = std::max(ax,bx);
        const double minY = std::min(ay,by), maxY = std::max(ay,by);
        const double px = ax + offset(random), py = ay + offset(random);
        const long double rx = std::max(0.L, std::max(static_cast<long double>(minX)-px, static_cast<long double>(px)-maxX));
        const long double ry = std::max(0.L, std::max(static_cast<long double>(minY)-py, static_cast<long double>(py)-maxY));
        require(static_cast<long double>(rectangleDistanceLower(px,py,minX,maxX,minY,maxY)) <= std::hypot(rx,ry), "Rectangle lower endpoint overestimates");
        ++rectangles;
    }
    XY xy{{0,0},{1,0},{1,1}};
    Graph g{{{1,.5}},{{0,.5},{2,.25}},{{1,.25}}};
    auto c = calibrate(xy,g,3);
    require(c.ready && c.scale > 0 && c.scale <= .25 && c.verification_violations == 0, "Calibration failed");
    require(!exceedsUpper(routeLower(c,rectangleDistanceLower(0,0,1,1,1,1),0),.75), "Feasible path pruned");
    require(!exceedsUpper(1.,1.) && exceedsUpper(2.,1.), "Pruning comparison failed");
    require(!exceedsUpper(1.,std::numeric_limits<double>::infinity()), "Infinite upper bound pruned");
    require(rectangleDistanceLower(1,1,1,1,1,1)==0, "Inside point must have zero bound");
    require(normUpper(1,1,1,1)==0, "Coincident points must have zero norm");
    auto zero = g; zero[0][0].second=0; zero[1][0].second=0;
    require(calibrate(xy,zero,3).scale==0, "Zero-weight nonzero-geometry edge must disable geometry");
    auto bad = g; bad[0][0].second=-1;
    mustReject([&]{calibrate(xy,bad,3);});
    bad=g;bad[0][0].first=3;mustReject([&]{calibrate(xy,bad,3);});
    auto invalid=xy;invalid[0].first=std::numeric_limits<double>::quiet_NaN();
    mustReject([&]{calibrate(invalid,g,3);});
    mustReject([&]{calibrate(xy,g,2);});
    mustReject([&]{routeLower(Certificate{},1,2);});
    mustReject([&]{rectangleDistanceLower(0,0,2,1,0,1);});
    mustReject([&]{exceedsUpper(1,-1);});
    const int prior=std::fegetround();std::fesetround(FE_UPWARD);
    mustReject([&]{calibrate(xy,g,3);});std::fesetround(prior);
    std::cout << "{\"passed\":true,\"edge_norm_interval_checks\":" << norms
              << ",\"rectangle_interval_checks\":" << rectangles
              << ",\"invalid_inputs_rejected\":8,\"zero_and_coincident_edges_checked\":true}" << std::endl;
}
