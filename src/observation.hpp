#pragma once
#include <algorithm>
#include <numeric>
#include "gaussian.hpp"

namespace ds {

struct Performer {
    Gaussian skill;
    double beta;
};

using Lineups = std::vector<std::vector<Performer>>;

// One observation: a single observation ranking several lineups. From the cavity
// priors it computes the observation's evidence and one likelihood message per
// performer.
class Observation {
public:
    double evidence = 1.0;
    std::vector<std::vector<Gaussian>> likelihoods;
    int sweeps = 0;
    static constexpr int kSweepCap = 256;
    // Sweeps run to convergence by default; set_sweeps(5, 0.0) restores a
    // fixed five-sweep budget.
    static inline int max_sweeps = 64;
    static inline double sweep_tol = 1e-12;
    static inline std::vector<long long> sweep_hist =
        std::vector<long long>(kSweepCap + 1, 0);

    Observation(const Lineups& lineups, const std::vector<double>& scores,
          bool continuous, double p_draw, double noise = 0.0,
          double p_chaos = 0.0)
        : lineups_(lineups), scores_(scores), continuous_(continuous),
          noise_(noise), p_chaos_(p_chaos) {
        size_t n = lineups.size();
        if (n < 2) throw std::invalid_argument("a observation needs at least 2 lineups");
        if (n != scores.size())
            throw std::invalid_argument("lineups and scores must match");
        for (const auto& lineup : lineups)
            if (lineup.empty()) throw std::invalid_argument("empty lineup");
        order_.resize(n);
        std::iota(order_.begin(), order_.end(), 0);
        // Stable, so tied scores keep input order and results are deterministic.
        std::stable_sort(order_.begin(), order_.end(),
                         [&](size_t a, size_t b) { return scores[a] > scores[b]; });
        team_prior_.reserve(n);
        for (size_t i = 0; i < n; ++i) team_prior_.push_back(team_performance(i));
        if (!continuous) {
            ties_.resize(n - 1);
            margins_.resize(n - 1);
            for (size_t e = 0; e + 1 < n; ++e) {
                ties_[e] = scores[order_[e]] == scores[order_[e + 1]];
                if (p_draw == 0.0) {
                    margins_[e] = 0.0;
                } else {
                    // The margin lives on the performance-difference scale,
                    // whose noise is the two lineups' betas combined.
                    double var = 0.0;
                    for (const auto& p : lineups[order_[e]]) var += p.beta * p.beta;
                    for (const auto& p : lineups[order_[e + 1]]) var += p.beta * p.beta;
                    margins_[e] = draw_margin(p_draw, std::sqrt(var));
                }
            }
        }
        likelihoods = n == 2 ? closed_form() : chain();
        if (p_chaos_ > 0.0 && !continuous_) apply_chaos();
    }

private:
    const Lineups& lineups_;
    const std::vector<double>& scores_;
    bool continuous_;
    double noise_;
    double p_chaos_;
    std::vector<size_t> order_;
    std::vector<Gaussian> team_prior_;
    std::vector<bool> ties_;
    std::vector<double> margins_;

    Gaussian team_performance(size_t i) const {
        Gaussian t{0.0, 0.0};
        for (const auto& p : lineups_[i])
            t = t + Gaussian(p.skill.mu,
                             std::sqrt(p.skill.sigma * p.skill.sigma + p.beta * p.beta));
        // Chair blur: the chair perceives team performance through excess
        // noise, inflating each team's variance.
        if (noise_ > 0.0)
            t = Gaussian(t.mu, std::sqrt(t.sigma * t.sigma + noise_));
        return t;
    }

    double observed_gap(size_t e) const {
        return scores_[order_[e]] - scores_[order_[e + 1]];
    }

    void accumulate_evidence(size_t e, const Gaussian& prior) {
        if (continuous_)
            evidence *= normal_pdf(observed_gap(e), prior.mu, prior.sigma);
        else if (ties_[e])
            evidence *= normal_cdf(margins_[e], prior.mu, prior.sigma) -
                        normal_cdf(-margins_[e], prior.mu, prior.sigma);
        else
            evidence *= 1.0 - normal_cdf(margins_[e], prior.mu, prior.sigma);
    }

    Gaussian gap_likelihood(size_t e, const Gaussian& prior) const {
        // A continuous gap is observed exactly: the message is a point mass.
        if (continuous_) return {observed_gap(e), 0.0};
        return truncate(prior, margins_[e], ties_[e]) / prior;
    }

    // Robust mixture: with probability p_chaos the observed ranking carries
    // no information; likelihood 1/n! (exact for strict rankings, reused as
    // an approximation for tied outcomes).
    void apply_chaos() {
        double z_chaos = 1.0;
        for (size_t i = 2; i <= lineups_.size(); ++i)
            z_chaos /= static_cast<double>(i);
        double clean = (1.0 - p_chaos_) * evidence;
        double total = clean + p_chaos_ * z_chaos;
        if (!(total > 0.0)) return;
        double w = clean / total;
        evidence = total;
        for (size_t t = 0; t < lineups_.size(); ++t) {
            for (size_t i = 0; i < lineups_[t].size(); ++i) {
                const Gaussian& prior = lineups_[t][i].skill;
                if (!(prior.sigma < INF)) continue;
                Gaussian post = prior * likelihoods[t][i];
                if (!(post.sigma < INF)) continue;
                // Moment-match the posterior mixture (clean posterior vs
                // untouched prior), then divide the prior back out.
                double m = w * post.mu + (1.0 - w) * prior.mu;
                double v = w * (post.sigma * post.sigma + post.mu * post.mu)
                         + (1.0 - w) * (prior.sigma * prior.sigma
                                        + prior.mu * prior.mu) - m * m;
                if (!(v > 0.0)) { likelihoods[t][i] = NINF; continue; }
                Gaussian mixed(m, std::sqrt(v));
                // A mixture no tighter than the prior has no valid Gaussian
                // likelihood; send the uninformative message instead.
                if (!(mixed.pi() - prior.pi() > 1e-12)) {
                    likelihoods[t][i] = NINF;
                    continue;
                }
                likelihoods[t][i] = mixed / prior;
            }
        }
    }

    // Two lineups: the single difference factor is solved in closed form.
    std::vector<std::vector<Gaussian>> closed_form() {
        size_t w = order_[0], l = order_[1];
        Gaussian d = team_prior_[w] - team_prior_[l];
        accumulate_evidence(0, d);
        Gaussian lik = gap_likelihood(0, d);
        std::vector<std::vector<Gaussian>> out(2);
        for (size_t rank = 0; rank < 2; ++rank) {
            size_t i = order_[rank];
            double sign = rank == 1 ? -1.0 : 1.0;
            out[i].reserve(lineups_[i].size());
            // The message to a performer must exclude their own prior, hence
            // the subtraction of that performer's skill variance.
            for (const auto& p : lineups_[i])
                out[i].emplace_back(
                    p.skill.mu + sign * (lik.mu - d.mu),
                    std::sqrt(lik.sigma * lik.sigma + d.sigma * d.sigma -
                              p.skill.sigma * p.skill.sigma));
        }
        return out;
    }

    static double update(Gaussian& slot, const Gaussian& next) {
        // An uninformative side means no comparable step yet: report INF so
        // the sweep loop cannot stop before real messages exist.
        if (!(slot.sigma < INF) || !(next.sigma < INF)) { slot = next; return INF; }
        auto d = next.delta(slot);
        slot = next;
        return std::max(d.first, d.second);
    }

    // Three or more lineups: the adjacent-difference factors form a chain,
    // so their messages are iterated (down the order, then back up).
    std::vector<std::vector<Gaussian>> chain() {
        size_t n = lineups_.size();
        std::vector<Gaussian> win(n, NINF), lose(n, NINF), dlik(n - 1, NINF);
        for (int it = 0; it < max_sweeps; ++it) {
            double moved = 0.0;
            for (size_t e = 0; e + 2 < n; ++e) {
                size_t a = order_[e], b = order_[e + 1];
                Gaussian dprior = team_prior_[a] * lose[a] - team_prior_[b] * win[b];
                // Evidence is taken on each edge's first visit, before its
                // message updates.
                if (it == 0) accumulate_evidence(e, dprior);
                moved = std::max(moved, update(dlik[e], gap_likelihood(e, dprior)));
                lose[b] = team_prior_[a] * lose[a] - dlik[e];
            }
            for (size_t e = n - 2; e >= 1; --e) {
                size_t a = order_[e], b = order_[e + 1];
                Gaussian dprior = team_prior_[a] * lose[a] - team_prior_[b] * win[b];
                if (it == 0 && e == n - 2) accumulate_evidence(e, dprior);
                moved = std::max(moved, update(dlik[e], gap_likelihood(e, dprior)));
                win[a] = team_prior_[b] * win[b] + dlik[e];
            }
            sweeps = it + 1;
            if (moved <= sweep_tol) break;
        }
        ++sweep_hist[sweeps];
        // The sweeps only maintain interior messages; fill in the endpoints.
        win[order_[0]] = team_prior_[order_[1]] * win[order_[1]] + dlik[0];
        lose[order_[n - 1]] = team_prior_[order_[n - 2]] * lose[order_[n - 2]] - dlik[n - 2];
        std::vector<std::vector<Gaussian>> out(n);
        for (size_t i = 0; i < n; ++i) {
            Gaussian team_lik = win[i] * lose[i];
            out[i].reserve(lineups_[i].size());
            // Team message -> performer message: subtract the teammates'
            // summed performance, then widen by beta to land on skill.
            for (const auto& p : lineups_[i]) {
                double pvar = p.skill.sigma * p.skill.sigma + p.beta * p.beta;
                Gaussian perf{p.skill.mu, std::sqrt(pvar)};
                Gaussian rest{team_prior_[i].mu - perf.mu,
                              std::sqrt(std::max(
                                  0.0, team_prior_[i].sigma * team_prior_[i].sigma - pvar))};
                Gaussian perf_lik = team_lik - rest;
                out[i].emplace_back(perf_lik.mu,
                                    std::sqrt(perf_lik.sigma * perf_lik.sigma +
                                              p.beta * p.beta));
            }
        }
        return out;
    }
};

}
