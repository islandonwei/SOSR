#ifndef SOSR_SAFE_GEOMETRY_H
#define SOSR_SAFE_GEOMETRY_H
#include <algorithm>
#include <cfenv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>
#ifdef __FAST_MATH__
#error "The geometric certificate must not be compiled with fast-math"
#endif

// Certificate for the immutable, loaded graph weights and projected coordinates.
// IEEE binary64, round-to-nearest, correctly rounded sqrt; compile without
// fast-math or floating-point contraction. All intermediate bounds are rounded
// outwards. This certifies geometry, not the correctness of a distance oracle.
namespace safe_geometry {
inline double up(double x) {
    return std::nextafter(x, std::numeric_limits<double>::infinity());
}
inline double downNonnegative(double x) {
    return x > 0.0 ? std::nextafter(x, 0.0) : 0.0;
}
inline void checkArithmetic() {
    if (!std::numeric_limits<double>::is_iec559 ||
        std::fegetround() != FE_TONEAREST)
        throw std::runtime_error("Geometry certificate requires IEEE binary64 round-to-nearest");
}
inline double normUpper(double ax, double ay, double bx, double by) {
    if (ax == bx && ay == by) return 0.0;
    const double dx = up(std::abs(ax - bx));
    const double dy = up(std::abs(ay - by));
    const double squared = up(up(dx * dx) + up(dy * dy));
    const double result = up(std::sqrt(squared));
    if (!std::isfinite(result))
        throw std::runtime_error("Non-finite projected edge distance");
    return result;
}

struct Certificate {
    bool ready = false;
    double scale = 0.0;
    std::size_t nodes = 0;
    std::uint64_t arcs = 0;
    std::uint64_t unscaled_overestimates = 0;
    std::uint64_t verification_violations = 0;
    double max_unscaled_ratio = 0.0;
    int limiting_u = -1;
    int limiting_v = -1;
};

template<class Adjacency>
Certificate calibrate(const std::vector<std::pair<double,double>>& xy,
                      const Adjacency& graph, int n) {
    checkArithmetic();
    if (n <= 0 || xy.size() != static_cast<std::size_t>(n))
        throw std::runtime_error("Graph/coordinate size mismatch");
    for (const auto& p : xy)
        if (!std::isfinite(p.first) || !std::isfinite(p.second))
            throw std::runtime_error("Non-finite projected coordinate");
    Certificate c;
    c.nodes = xy.size();
    c.scale = 1.0;
    for (int u = 0; u < n; ++u) {
        for (const auto& edge : graph[u]) {
            const int v = edge.first;
            const double w = edge.second;
            if (v < 0 || v >= n || !std::isfinite(w) || w < 0.0)
                throw std::runtime_error("Invalid edge in geometry calibration");
            ++c.arcs;
            const double length = normUpper(xy[u].first, xy[u].second,
                                            xy[v].first, xy[v].second);
            if (length == 0.0) continue;
            const double raw = std::hypot(xy[u].first - xy[v].first,
                                           xy[u].second - xy[v].second);
            c.unscaled_overestimates += raw > w + 1e-6;
            if (w > 0.0) c.max_unscaled_ratio = std::max(c.max_unscaled_ratio, raw / w);
            // Lower endpoint for w / upper_endpoint(norm). Lowering w as well
            // leaves room for the upper-rounded multiplication in verification.
            const double ratio = downNonnegative(downNonnegative(w) / length);
            if (ratio < c.scale) {
                c.scale = ratio;
                c.limiting_u = u;
                c.limiting_v = v;
            }
        }
    }
    c.scale = downNonnegative(c.scale);
    // Independent second pass, no tolerance: every upper-rounded product must
    // be <= the actual stored edge weight. Failure stops initialization.
    for (int u = 0; u < n; ++u) {
        for (const auto& edge : graph[u]) {
            const auto& p = xy[edge.first];
            const double length = normUpper(xy[u].first, xy[u].second, p.first, p.second);
            const double bound = c.scale == 0.0 || length == 0.0 ? 0.0 : up(c.scale * length);
            c.verification_violations += bound > edge.second;
        }
    }
    if (c.verification_violations)
        throw std::runtime_error("Geometric edge certificate failed; refusing geometric pruning");
    c.ready = true;
    return c;
}

inline double rectangleDistanceLower(double px, double py, double minX,
                                     double maxX, double minY, double maxY) {
    if (!std::isfinite(px) || !std::isfinite(py) || !std::isfinite(minX) ||
        !std::isfinite(maxX) || !std::isfinite(minY) || !std::isfinite(maxY) ||
        minX > maxX || minY > maxY)
        throw std::runtime_error("Invalid rectangle distance input");
    const double dx = std::max(downNonnegative(minX - px), downNonnegative(px - maxX));
    const double dy = std::max(downNonnegative(minY - py), downNonnegative(py - maxY));
    const double squared = downNonnegative(downNonnegative(dx * dx) + downNonnegative(dy * dy));
    if (!std::isfinite(squared))
        throw std::runtime_error("Rectangle distance overflow");
    return downNonnegative(std::sqrt(squared));
}

inline double routeLower(const Certificate& c, double sourceLower, double targetLower) {
    if (!c.ready) throw std::runtime_error("Geometric pruning used before calibration");
    if (!std::isfinite(c.scale) || c.scale < 0.0 || c.scale > 1.0 ||
        !std::isfinite(sourceLower) || !std::isfinite(targetLower) ||
        sourceLower < 0.0 || targetLower < 0.0)
        throw std::runtime_error("Invalid geometric certificate or distance bound");
    return downNonnegative(c.scale * downNonnegative(sourceLower + targetLower));
}
inline bool exceedsUpper(double lower, double upper) {
    if (!std::isfinite(lower) || lower < 0.0 || std::isnan(upper) || upper < 0.0)
        throw std::runtime_error("Invalid pruning bound");
    if (!std::isfinite(upper)) return false;
    // Separate numerical comparison slack; it is NOT the admissibility fix.
    const double slack = 1e-6 + 1e-9 * std::max(1.0, std::abs(upper));
    return lower > up(upper + slack);
}
}
#endif
