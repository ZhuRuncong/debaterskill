#pragma once
#include <cstdint>
#include <cstdio>
#include <string>
#include <unordered_map>
#include "ballot.hpp"

namespace ds {

struct Entity {
    Gaussian prior;
    double beta;
    double gamma;
};

struct BallotSpec {
    int day;
    std::vector<std::vector<uint32_t>> lineups;
    std::vector<double> scores;
    bool continuous;
    double p_draw;
    int tag;
    double noise;
    double p_chaos;
};

// The whole record: interns entities, batches observations by time period,
// and runs forward-backward EP smoothing over the skill chains.
class Tab {
public:
    Tab(double mu, double sigma, double beta, double gamma, int period_days,
        double p_chaos = 0.0)
        : mu0_(mu), sigma0_(sigma), beta0_(beta), gamma0_(gamma),
          period_(period_days < 1 ? 1 : period_days), p_chaos_(p_chaos) {}

    void set_p_chaos(double v) {
        if (v < 0.0 || v >= 1.0)
            throw std::invalid_argument("p_chaos must be in [0, 1)");
        p_chaos_ = v;
    }
    double p_chaos() const { return p_chaos_; }

    uint32_t intern(const std::string& key) {
        auto it = ids_.find(key);
        if (it != ids_.end()) return it->second;
        uint32_t id = static_cast<uint32_t>(entities_.size());
        ids_.emplace(key, id);
        keys_.push_back(key);
        entities_.push_back({Gaussian(mu0_, sigma0_), beta0_, gamma0_});
        return id;
    }

    void enroll(const std::string& key, double mu, double sigma, double beta, double gamma) {
        entities_[intern(key)] = {Gaussian(mu, sigma), beta, gamma};
    }

    void add_ballot(int day, const std::vector<std::vector<std::string>>& lineups,
                   const std::vector<double>& scores, bool continuous, double p_draw,
                   int tag, double noise) {
        if (lineups.size() < 2) throw std::invalid_argument("need at least 2 lineups");
        if (lineups.size() != scores.size())
            throw std::invalid_argument("lineups and scores must match");
        if (noise < 0.0) throw std::invalid_argument("noise must be >= 0");
        // Chaos is a mixture over rankings; a continuous observation has no
        // ranking to scramble, so it stays clean.
        BallotSpec spec{day, {}, scores, continuous, p_draw, tag, noise,
                       continuous ? 0.0 : p_chaos_};
        for (const auto& lineup : lineups) {
            if (lineup.empty()) throw std::invalid_argument("empty lineup");
            spec.lineups.emplace_back();
            for (const auto& name : lineup) spec.lineups.back().push_back(intern(name));
        }
        specs_.push_back(std::move(spec));
    }

    size_t size() const { return specs_.size(); }
    size_t n_entities() const { return entities_.size(); }
    const std::string& key(uint32_t id) const { return keys_[id]; }

    void set_noise(size_t i, double noise) {
        if (noise < 0.0) throw std::invalid_argument("noise must be >= 0");
        specs_.at(i).noise = noise;
    }

    // Leave-one-out log evidence of ballot i under a candidate noise, scored
    // against the fitted state with i's own messages divided out.
    double evidence_at(size_t i, double noise) const {
        if (batches_.empty()) throw std::runtime_error("fit before evidence_at");
        auto [b, e] = where_.at(i);
        const Batch& batch = batches_[b];
        const BallotSpec& spec = specs_[i];
        Lineups lu = cavity(batch, batch.events[e]);
        Ballot c(lu, spec.scores, spec.continuous, spec.p_draw, noise, spec.p_chaos);
        return std::log(std::max(c.evidence, 1e-300));
    }

    std::pair<std::pair<double, double>, int> fit(int iterations, double epsilon,
                                                  bool verbose) {
        build();
        std::pair<double, double> step{INF, INF};
        int i = 0;
        while (i < iterations && std::max(step.first, step.second) > epsilon) {
            step = iteration();
            ++i;
            if (verbose)
                std::printf("iteration %d step (%.6g, %.6g)\n", i, step.first, step.second);
        }
        finalize();
        return {step, i};
    }

    double log_evidence(int tag) const {
        double total = 0.0;
        // Zero-evidence events would contribute -inf: outcomes impossible
        // under the model (a tie at p_draw=0) or whose probability underflows
        // (upsets past z~8). Excluded here, reported by evidence_counts.
        for (const auto& b : batches_)
            for (size_t e = 0; e < b.events.size(); ++e)
                if (b.evid[e] > 0.0 &&
                    (tag < 0 || specs_[b.events[e].spec].tag == tag))
                    total += std::log(b.evid[e]);
        return total;
    }

    std::pair<int, int> evidence_counts(int tag) const {
        int used = 0, dropped = 0;
        for (const auto& b : batches_)
            for (size_t e = 0; e < b.events.size(); ++e)
                if (tag < 0 || specs_[b.events[e].spec].tag == tag)
                    (b.evid[e] > 0.0 ? used : dropped) += 1;
        return {used, dropped};
    }

    std::vector<std::pair<uint32_t, std::vector<std::array<double, 3>>>> curves() const {
        std::vector<std::vector<std::array<double, 3>>> acc(entities_.size());
        for (const auto& b : batches_)
            for (const auto& node : b.skills) {
                Gaussian p = posterior(node);
                acc[node.entity].push_back({static_cast<double>(b.time), p.mu, p.sigma});
            }
        std::vector<std::pair<uint32_t, std::vector<std::array<double, 3>>>> out;
        for (uint32_t id = 0; id < acc.size(); ++id)
            if (!acc[id].empty()) out.emplace_back(id, std::move(acc[id]));
        return out;
    }

    double forecast(int day, const std::vector<std::vector<std::string>>& lineups,
                    const std::vector<double>& scores, bool continuous,
                    double p_draw, double noise) const {
        int pday = period_time(day);
        Lineups lu;
        for (const auto& lineup : lineups) {
            lu.emplace_back();
            for (const auto& name : lineup) {
                auto it = ids_.find(name);
                // Unknown names are legal by design: held-out rooms contain
                // debut speakers, priced at the raw prior rather than raising.
                if (it == ids_.end()) {
                    lu.back().push_back({Gaussian(mu0_, sigma0_), beta0_});
                    continue;
                }
                const Entity& ent = entities_[it->second];
                auto st = final_.find(it->second);
                if (st == final_.end()) {
                    lu.back().push_back({ent.prior, ent.beta});
                } else {
                    double elapsed = std::fabs(static_cast<double>(pday - st->second.first));
                    double drift =
                        std::min(std::sqrt(elapsed) * ent.gamma, 1.67 * sigma0_);
                    lu.back().push_back({st->second.second + Gaussian(0.0, drift), ent.beta});
                }
            }
        }
        Ballot c(lu, scores, continuous, p_draw, noise,
                continuous ? 0.0 : p_chaos_);
        return std::log(std::max(c.evidence, 1e-300));
    }

private:
    struct Slot {
        uint32_t member;
        uint32_t slot;
    };
    struct Event {
        uint32_t spec;
        std::vector<std::vector<Slot>> slots;
    };
    struct SkillNode {
        uint32_t entity;
        Gaussian forward = NINF;
        Gaussian backward = NINF;
        // One slot per appearance within the batch, so an entity playing
        // several games in one period gets independent likelihood messages.
        std::vector<Gaussian> lik;
    };
    struct Batch {
        int time;
        std::vector<SkillNode> skills;
        std::unordered_map<uint32_t, uint32_t> index;
        std::vector<Event> events;
        std::vector<double> evid;
    };

    double mu0_, sigma0_, beta0_, gamma0_;
    int period_;
    double p_chaos_ = 0.0;
    std::unordered_map<std::string, uint32_t> ids_;
    std::vector<std::string> keys_;
    std::vector<Entity> entities_;
    std::vector<BallotSpec> specs_;
    std::vector<Batch> batches_;
    std::vector<std::pair<uint32_t, uint32_t>> where_;
    std::unordered_map<uint32_t, std::pair<int, Gaussian>> final_;

    // Floor to the period start; C++ % truncates toward zero, so negative
    // days need the correction to keep flooring.
    int period_time(int day) const {
        int r = day % period_;
        if (r < 0) r += period_;
        return day - r;
    }

    // What the past says (forward), what the future says (backward), and
    // what that day's rooms said (lik).
    static Gaussian posterior(const SkillNode& node) {
        return node.forward * node.backward * product(node.lik);
    }

    void build() {
        batches_.clear();
        final_.clear();
        where_.assign(specs_.size(), {0, 0});
        std::vector<uint32_t> idx(specs_.size());
        std::iota(idx.begin(), idx.end(), 0);
        // Stable, so events within one period keep insertion order and
        // repeated fits are deterministic.
        std::stable_sort(idx.begin(), idx.end(), [&](uint32_t a, uint32_t b) {
            return period_time(specs_[a].day) < period_time(specs_[b].day);
        });
        for (uint32_t si : idx) {
            int t = period_time(specs_[si].day);
            if (batches_.empty() || batches_.back().time != t) {
                batches_.emplace_back();
                batches_.back().time = t;
            }
            Batch& b = batches_.back();
            Event ev{si, {}};
            uint32_t e = static_cast<uint32_t>(b.events.size());
            where_[si] = {static_cast<uint32_t>(batches_.size() - 1), e};
            for (const auto& lineup : specs_[si].lineups) {
                ev.slots.emplace_back();
                for (uint32_t id : lineup) {
                    auto it = b.index.find(id);
                    uint32_t m;
                    if (it == b.index.end()) {
                        m = static_cast<uint32_t>(b.skills.size());
                        b.index.emplace(id, m);
                        b.skills.push_back(SkillNode{id});
                    } else {
                        m = it->second;
                    }
                    ev.slots.back().push_back(
                        {m, static_cast<uint32_t>(b.skills[m].lik.size())});
                    b.skills[m].lik.push_back(NINF);
                }
            }
            b.events.push_back(std::move(ev));
            b.evid.push_back(0.0);
        }
    }

    // EP cavity: the posterior with this appearance's own message divided
    // out. Per appearance, so an entity seated twice in one event keeps its
    // sibling appearance's message.
    Lineups cavity(const Batch& b, const Event& ev) const {
        Lineups lu(ev.slots.size());
        for (size_t t = 0; t < ev.slots.size(); ++t)
            for (const Slot& s : ev.slots[t]) {
                const SkillNode& node = b.skills[s.member];
                lu[t].push_back({posterior(node) / node.lik[s.slot],
                                 entities_[node.entity].beta});
            }
        return lu;
    }

    std::pair<double, double> refresh(Batch& b, size_t e) {
        const Event& ev = b.events[e];
        const BallotSpec& spec = specs_[ev.spec];
        Lineups lu = cavity(b, ev);
        Ballot c(lu, spec.scores, spec.continuous, spec.p_draw, spec.noise,
                spec.p_chaos);
        b.evid[e] = c.evidence;
        std::pair<double, double> step{0.0, 0.0};
        for (size_t t = 0; t < ev.slots.size(); ++t)
            for (size_t j = 0; j < ev.slots[t].size(); ++j) {
                const Slot& s = ev.slots[t][j];
                SkillNode& node = b.skills[s.member];
                auto d = c.likelihoods[t][j].delta(node.lik[s.slot]);
                step.first = std::max(step.first, d.first);
                step.second = std::max(step.second, d.second);
                node.lik[s.slot] = c.likelihoods[t][j];
            }
        return step;
    }

    Gaussian receive(const SkillNode& node, int now,
                     const std::unordered_map<uint32_t, Gaussian>& msg,
                     const std::unordered_map<uint32_t, int>& when) const {
        const Gaussian& m = msg.at(node.entity);
        double elapsed = std::fabs(static_cast<double>(now - when.at(node.entity)));
        // The added drift SD is capped at 1.67*sigma0 so the inflation from
        // a single absence is bounded.
        double drift =
            std::min(std::sqrt(elapsed) * entities_[node.entity].gamma, 1.67 * sigma0_);
        return m + Gaussian(0.0, drift);
    }

    std::pair<double, double> forward_pass() {
        std::unordered_map<uint32_t, Gaussian> msg;
        std::unordered_map<uint32_t, int> when;
        std::pair<double, double> step{0.0, 0.0};
        for (Batch& b : batches_) {
            for (SkillNode& node : b.skills)
                node.forward = msg.count(node.entity)
                                   ? receive(node, b.time, msg, when)
                                   : entities_[node.entity].prior + Gaussian(0.0, 0.0);
            for (size_t e = 0; e < b.events.size(); ++e) {
                auto s = refresh(b, e);
                step.first = std::max(step.first, s.first);
                step.second = std::max(step.second, s.second);
            }
            for (SkillNode& node : b.skills) {
                msg[node.entity] = node.forward * product(node.lik);
                when[node.entity] = b.time;
            }
        }
        return step;
    }

    std::pair<double, double> backward_pass() {
        std::unordered_map<uint32_t, Gaussian> msg;
        std::unordered_map<uint32_t, int> when;
        std::pair<double, double> step{0.0, 0.0};
        // Nothing lies after the last batch: its outgoing message is seeded
        // from its own state and the batch itself is not refreshed here.
        Batch& last = batches_.back();
        for (SkillNode& node : last.skills) {
            msg[node.entity] = node.backward * product(node.lik);
            when[node.entity] = last.time;
        }
        for (size_t bi = batches_.size() - 1; bi-- > 0;) {
            Batch& b = batches_[bi];
            for (SkillNode& node : b.skills)
                if (msg.count(node.entity)) node.backward = receive(node, b.time, msg, when);
            for (size_t e = 0; e < b.events.size(); ++e) {
                auto s = refresh(b, e);
                step.first = std::max(step.first, s.first);
                step.second = std::max(step.second, s.second);
            }
            for (SkillNode& node : b.skills) {
                msg[node.entity] = node.backward * product(node.lik);
                when[node.entity] = b.time;
            }
        }
        return step;
    }

    std::pair<double, double> iteration() {
        if (batches_.empty()) return {0.0, 0.0};
        auto f = forward_pass();
        auto bk = backward_pass();
        return {std::max(f.first, bk.first), std::max(f.second, bk.second)};
    }

    // Last posterior per entity, kept as the drift origin for forecast().
    void finalize() {
        for (const auto& b : batches_)
            for (const auto& node : b.skills)
                final_[node.entity] = {b.time, posterior(node)};
    }
};

}
