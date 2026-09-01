#pragma once
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ds {

inline constexpr double INF = std::numeric_limits<double>::infinity();
inline const double SQRT2 = std::sqrt(2.0);
inline const double SQRT2PI = std::sqrt(2.0 * 3.14159265358979323846);

// One-dimensional Gaussian in an EP message role: sigma = INF is the
// uninformative message, * / are natural-parameter product and division.
struct Gaussian {
    double mu = 0.0;
    double sigma = INF;

    Gaussian() = default;
    Gaussian(double m, double s) : mu(m), sigma(s) {
        if (!(s >= 0.0)) throw std::domain_error("sigma must be >= 0");
    }

    double pi() const { return sigma > 0.0 ? 1.0 / (sigma * sigma) : INF; }
    double tau() const { return sigma > 0.0 ? mu / (sigma * sigma) : 0.0; }

    Gaussian operator+(const Gaussian& o) const {
        return {mu + o.mu, std::sqrt(sigma * sigma + o.sigma * o.sigma)};
    }

    Gaussian operator-(const Gaussian& o) const {
        return {mu - o.mu, std::sqrt(sigma * sigma + o.sigma * o.sigma)};
    }

    Gaussian operator*(const Gaussian& o) const {
        if (o.pi() == 0.0) return *this;
        if (pi() == 0.0) return o;
        double p = pi() + o.pi();
        return {(tau() + o.tau()) / p, 1.0 / std::sqrt(p)};
    }

    Gaussian operator/(const Gaussian& o) const {
        double p = pi() - o.pi();
        // Equal precisions make the ratio improper; collapse to uninformative.
        if (p == 0.0) return {0.0, INF};
        return {(tau() - o.tau()) / p, 1.0 / std::sqrt(p)};
    }

    std::pair<double, double> delta(const Gaussian& o) const {
        return {std::fabs(mu - o.mu), std::fabs(sigma - o.sigma)};
    }
};

inline const Gaussian NINF{0.0, INF};

inline Gaussian product(const std::vector<Gaussian>& gs) {
    Gaussian res = NINF;
    for (const auto& g : gs) res = res * g;
    return res;
}

inline double normal_cdf(double x, double mu, double sigma) {
    return 0.5 * std::erfc((mu - x) / (sigma * SQRT2));
}

inline double normal_pdf(double x, double mu, double sigma) {
    double z = x - mu;
    return std::exp(-z * z / (2.0 * sigma * sigma)) / (SQRT2PI * sigma);
}

// Rational approximation, then one refinement step.
inline double normal_inv_cdf(double p) {
    static const double a[] = {-3.969683028665376e+01, 2.209460984245205e+02,
                               -2.759285104469687e+02, 1.383577518672690e+02,
                               -3.066479806614716e+01, 2.506628277459239e+00};
    static const double b[] = {-5.447609879822406e+01, 1.615858368580409e+02,
                               -1.556989798598866e+02, 6.680131188771972e+01,
                               -1.328068155288572e+01};
    static const double c[] = {-7.784894002430293e-03, -3.223964580411365e-01,
                               -2.400758277161838e+00, -2.549732539343734e+00,
                               4.374664141464968e+00,  2.938163982698783e+00};
    static const double d[] = {7.784695709041462e-03, 3.224671290700398e-01,
                               2.445134137142996e+00, 3.754408661907416e+00};
    double x;
    if (p < 0.02425) {
        double q = std::sqrt(-2.0 * std::log(p));
        x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    } else if (p <= 0.97575) {
        double q = p - 0.5, r = q * q;
        x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
            (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0);
    } else {
        double q = std::sqrt(-2.0 * std::log(1.0 - p));
        x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    }
    double e = 0.5 * std::erfc(-x / SQRT2) - p;
    double u = e * SQRT2PI * std::exp(x * x / 2.0);
    return x - u / (1.0 + x * u / 2.0);
}

// Margin whose window under N(0, sd) has probability p_draw.
inline double draw_margin(double p_draw, double sd) {
    return std::fabs(normal_inv_cdf(0.5 - p_draw / 2.0) * sd);
}

// The difference truncated to the win region (d > margin) or the draw window
// (|d| <= margin), replaced by the moment-matched Gaussian. v is the mean
// correction weighted by surprise; w the fraction of variance removed.
inline Gaussian truncate(const Gaussian& prior, double margin, bool tie) {
    double mu = prior.mu, sigma = prior.sigma, v, w;
    if (!tie) {
        double alpha = (margin - mu) / sigma;
        v = normal_pdf(-alpha, 0.0, 1.0) / normal_cdf(-alpha, 0.0, 1.0);
        w = v * (v - alpha);
    } else {
        double lo = (-margin - mu) / sigma, hi = (margin - mu) / sigma;
        double den = normal_cdf(hi, 0.0, 1.0) - normal_cdf(lo, 0.0, 1.0);
        v = (normal_pdf(lo, 0.0, 1.0) - normal_pdf(hi, 0.0, 1.0)) / den;
        double u = (lo * normal_pdf(lo, 0.0, 1.0) - hi * normal_pdf(hi, 0.0, 1.0)) / den;
        w = v * v - u;
    }
    return {mu + sigma * v, sigma * std::sqrt(1.0 - w)};
}

}
