#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "tab.hpp"

namespace py = pybind11;
using namespace ds;

PYBIND11_MODULE(_core, m) {
    m.def("sweep_hist", []() { return Ballot::sweep_hist; });
    m.def("reset_sweep_hist", []() {
        std::fill(Ballot::sweep_hist.begin(), Ballot::sweep_hist.end(), 0LL);
    });
    m.def("set_sweeps", [](int max_sweeps, double tol) {
        if (max_sweeps < 1 || max_sweeps > Ballot::kSweepCap)
            throw std::invalid_argument("max_sweeps out of range");
        Ballot::max_sweeps = max_sweeps;
        Ballot::sweep_tol = tol;
    }, py::arg("max_sweeps"), py::arg("tol"));
    m.def("get_sweeps", []() {
        return py::make_tuple(Ballot::max_sweeps, Ballot::sweep_tol);
    });

    py::class_<Gaussian>(m, "Gaussian")
        .def(py::init<double, double>(), py::arg("mu") = 0.0, py::arg("sigma") = INF)
        .def_readonly("mu", &Gaussian::mu)
        .def_readonly("sigma", &Gaussian::sigma)
        .def("__add__", &Gaussian::operator+)
        .def("__sub__", &Gaussian::operator-)
        .def("__mul__",
             static_cast<Gaussian (Gaussian::*)(const Gaussian&) const>(
                 &Gaussian::operator*))
        .def("__truediv__", &Gaussian::operator/)
        .def("__iter__",
             [](const Gaussian& g) {
                 return py::iter(py::make_tuple(g.mu, g.sigma));
             })
        .def("__repr__", [](const Gaussian& g) {
            return "N(mu=" + std::to_string(g.mu) + ", sigma=" + std::to_string(g.sigma) +
                   ")";
        });

    m.def(
        "ballot",
        [](const std::vector<std::vector<std::tuple<double, double, double>>>& lineups,
           const std::vector<double>& scores, bool continuous, double p_draw,
           double p_chaos) {
            Lineups lu;
            for (const auto& lineup : lineups) {
                lu.emplace_back();
                for (const auto& [mu, sigma, beta] : lineup)
                    lu.back().push_back({Gaussian(mu, sigma), beta});
            }
            Ballot c(lu, scores, continuous, p_draw, 0.0, p_chaos);
            std::vector<std::vector<std::pair<double, double>>> lik;
            for (const auto& row : c.likelihoods) {
                lik.emplace_back();
                for (const auto& g : row) lik.back().emplace_back(g.mu, g.sigma);
            }
            return py::make_tuple(c.evidence, lik);
        },
        py::arg("lineups"), py::arg("scores"), py::arg("continuous") = false,
        py::arg("p_draw") = 0.0, py::arg("p_chaos") = 0.0);

    py::class_<Tab>(m, "TabCore")
        .def(py::init<double, double, double, double, int, double>(), py::arg("mu"),
             py::arg("sigma"), py::arg("beta"), py::arg("gamma"),
             py::arg("period_days"), py::arg("p_chaos") = 0.0)
        .def("set_p_chaos", &Tab::set_p_chaos)
        .def("enroll", &Tab::enroll)
        .def("add_ballot", &Tab::add_ballot, py::arg("day"), py::arg("lineups"),
             py::arg("scores"), py::arg("continuous") = false, py::arg("p_draw") = 0.0,
             py::arg("tag") = 3, py::arg("noise") = 0.0)
        .def("set_noise", &Tab::set_noise, py::arg("i"), py::arg("noise"))
        .def("evidence_at", &Tab::evidence_at, py::arg("i"), py::arg("noise"))
        // fit runs for minutes on a full corpus; release the GIL for it.
        .def("fit", &Tab::fit, py::arg("iterations") = 10, py::arg("epsilon") = 1e-4,
             py::arg("verbose") = false, py::call_guard<py::gil_scoped_release>())
        .def("log_evidence", &Tab::log_evidence, py::arg("tag") = -1)
        .def("evidence_counts", &Tab::evidence_counts, py::arg("tag") = -1)
        .def("curves",
             [](const Tab& t) {
                 py::dict out;
                 for (const auto& [id, pts] : t.curves()) {
                     py::list l;
                     for (const auto& p : pts)
                         l.append(py::make_tuple(static_cast<int>(p[0]), p[1], p[2]));
                     out[py::str(t.key(id))] = l;
                 }
                 return out;
             })
        .def("forecast", &Tab::forecast, py::arg("day"), py::arg("lineups"),
             py::arg("scores"), py::arg("continuous") = false, py::arg("p_draw") = 0.0,
             py::arg("noise") = 0.0)
        .def("size", &Tab::size)
        .def("n_entities", &Tab::n_entities);
}
