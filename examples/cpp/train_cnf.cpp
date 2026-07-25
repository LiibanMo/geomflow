#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <functional>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <geomflow/adjoint.h>
#include <geomflow/integrator.h>
#include <geomflow/manifold.h>
#include <geomflow/metric.h>
#include <geomflow/tangent.h>
#include <geomflow/vector_field.h>

#include "manifolds/sphere.h"
#include "manifolds/torus.h"

using namespace geomflow;
using Traits = ManifoldTraits<2>;
using Scalar = Traits::ScalarType;
using Point = Traits::Point;
using Tangent = TangentVector<Traits>;
using Cotangent = CotangentVector<Traits>;

enum class ManifoldType { Euclidean, Sphere, Torus };

struct Config {
  ManifoldType manifold = ManifoldType::Euclidean;
  size_t epochs = 200;
  size_t batch_size = 256;
  size_t hidden_dim = 32;
  Scalar lr = 0.01;
  Scalar dt = 0.05;
  Scalar t0 = 0;
  Scalar t1 = 1;
  std::string output_model = "cnf_model.bin";
};

struct MLPArch {
  size_t W1_size;
  size_t b1_size;
  size_t W2_size;
  size_t b2_size;
  size_t total;
};

MLPArch mlp_arch(size_t input_dim, size_t hidden_dim, size_t output_dim) {
  size_t W1 = hidden_dim * input_dim;
  size_t b1 = hidden_dim;
  size_t W2 = output_dim * hidden_dim;
  size_t b2 = output_dim;
  return {W1, b1, W2, b2, W1 + b1 + W2 + b2};
}

constexpr size_t N = 2;
constexpr size_t HIDDEN = 32;
constexpr size_t INPUT = N + 1;
constexpr size_t OUTPUT = N;

auto make_mlp_forward() {
  auto arch = mlp_arch(INPUT, HIDDEN, OUTPUT);
  return [arch](Scalar t, const Point& x, const std::vector<Scalar>& theta) -> Tangent {
    Scalar input[INPUT];
    input[0] = t;
    for (size_t i = 0; i < N; ++i)
      input[i + 1] = x[i];

    Scalar hidden[HIDDEN];
    for (size_t i = 0; i < HIDDEN; ++i) {
      Scalar s = theta[arch.W1_size + i];
      for (size_t j = 0; j < INPUT; ++j)
        s += theta[i * INPUT + j] * input[j];
      hidden[i] = std::tanh(s);
    }

    Tangent result{};
    size_t offset = arch.W1_size + arch.b1_size;
    for (size_t i = 0; i < OUTPUT; ++i) {
      Scalar s = theta[offset + arch.W2_size + i];
      for (size_t j = 0; j < HIDDEN; ++j)
        s += theta[offset + i * HIDDEN + j] * hidden[j];
      result.components[i] = s;
    }
    return result;
  };
}

std::vector<Scalar> init_params(const MLPArch& arch) {
  std::mt19937 rng(42);
  std::normal_distribution<Scalar> dist(0, 0.1);
  std::vector<Scalar> params(arch.total);
  for (auto& p : params)
    p = dist(rng);
  return params;
}

std::vector<Point> generate_target_euclidean(size_t n) {
  std::mt19937 rng(123);
  std::vector<Point> data;
  data.reserve(n);
  std::array<Point, 4> centers = {{{3, 3}, {3, -3}, {-3, 3}, {-3, -3}}};
  std::normal_distribution<Scalar> noise(0, 0.3);
  std::uniform_int_distribution<int> pick(0, 3);

  for (size_t i = 0; i < n; ++i) {
    int c = pick(rng);
    Point p;
    p[0] = centers[c][0] + noise(rng);
    p[1] = centers[c][1] + noise(rng);
    data.push_back(p);
  }
  return data;
}

std::vector<Point> generate_target_sphere(size_t n) {
  std::mt19937 rng(123);
  std::vector<Point> data;
  data.reserve(n);
  std::normal_distribution<Scalar> theta_dist(1.57, 0.3); // near equator
  std::uniform_real_distribution<Scalar> phi_dist(0, 2 * M_PI);
  std::normal_distribution<Scalar> noise(0, 0.05);

  for (size_t i = 0; i < n; ++i) {
    Point p;
    p[0] = theta_dist(rng) + noise(rng);
    p[1] = phi_dist(rng) + noise(rng);
    data.push_back(p);
  }
  return data;
}

std::vector<Point> generate_target_torus(size_t n) {
  std::mt19937 rng(123);
  std::vector<Point> data;
  data.reserve(n);
  std::uniform_real_distribution<Scalar> theta_dist(0, 2 * M_PI);
  std::normal_distribution<Scalar> phi_dist1(0, 0.3);
  std::normal_distribution<Scalar> phi_dist2(M_PI, 0.3);
  std::uniform_int_distribution<int> pick(0, 1);
  std::normal_distribution<Scalar> noise(0, 0.1);

  for (size_t i = 0; i < n; ++i) {
    Point p;
    p[0] = theta_dist(rng) + noise(rng);
    p[1] = (pick(rng) == 0) ? phi_dist1(rng) + noise(rng) : phi_dist2(rng) + noise(rng);
    data.push_back(p);
  }
  return data;
}

Scalar log_target_density(const Point& x, ManifoldType mt) {
  auto gauss_log = [](const Point& x, const Point& mu, Scalar sigma) -> Scalar {
    Scalar dx0 = x[0] - mu[0];
    Scalar dx1 = x[1] - mu[1];
    Scalar s2 = sigma * sigma;
    return -(dx0 * dx0 + dx1 * dx1) / (2 * s2) - std::log(2 * M_PI * s2);
  };

  auto log_sum_exp = [](Scalar a, Scalar b) -> Scalar {
    Scalar m = std::max(a, b);
    return m + std::log(std::exp(a - m) + std::exp(b - m));
  };

  if (mt == ManifoldType::Euclidean) {
    std::array<Point, 4> mus = {{{3, 3}, {3, -3}, {-3, 3}, {-3, -3}}};
    Scalar lp = gauss_log(x, mus[0], 0.3);
    for (size_t i = 1; i < 4; ++i)
      lp = log_sum_exp(lp, gauss_log(x, mus[i], 0.3));
    return lp - std::log(Scalar(4));
  } else if (mt == ManifoldType::Sphere) {
    std::array<Point, 2> mus = {{{1.57, 0.5}, {1.57, 3.64}}};
    Scalar lp = log_sum_exp(gauss_log(x, mus[0], 0.3), gauss_log(x, mus[1], 0.3));
    return lp - std::log(Scalar(2));
  } else {
    // Torus: two rings in phi direction
    Scalar px = std::cos(x[1]);
    Scalar py = std::sin(x[1]);
    Point mu1 = {x[0], 0};
    Point mu2 = {x[0], Scalar(M_PI)};
    Scalar lp = log_sum_exp(gauss_log({px, py}, {1, 0}, 0.3),
                            gauss_log({px, py}, {-1, 0}, 0.3));
    return lp - std::log(Scalar(2));
  }
}

Tangent grad_log_target(const Point& x, ManifoldType mt, Scalar h = 1e-4) {
  Tangent grad{};
  for (size_t i = 0; i < N; ++i) {
    Point xp = x, xm = x;
    xp[i] += h;
    xm[i] -= h;
    grad.components[i] = (log_target_density(xp, mt) - log_target_density(xm, mt)) / (2 * h);
  }
  return grad;
}

void save_params(const std::string& path, const std::vector<Scalar>& params, ManifoldType mt) {
  std::ofstream f(path, std::ios::binary);
  int mt_int = static_cast<int>(mt);
  f.write(reinterpret_cast<const char*>(&mt_int), sizeof(int));
  size_t sz = params.size();
  f.write(reinterpret_cast<const char*>(&sz), sizeof(size_t));
  f.write(reinterpret_cast<const char*>(params.data()), sz * sizeof(Scalar));
}

void print_usage(const char* prog) {
  std::cerr << "Usage: " << prog
            << " --manifold <euclidean|sphere|torus> [--epochs 200] [--batch 256] "
               "[--lr 0.01] [--output cnf_model.bin]\n";
}

int main(int argc, char** argv) {
  Config cfg;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--manifold" && i + 1 < argc) {
      std::string v = argv[++i];
      if (v == "euclidean")
        cfg.manifold = ManifoldType::Euclidean;
      else if (v == "sphere")
        cfg.manifold = ManifoldType::Sphere;
      else if (v == "torus")
        cfg.manifold = ManifoldType::Torus;
      else {
        std::cerr << "Unknown manifold: " << v << "\n";
        print_usage(argv[0]);
        return 1;
      }
    } else if (arg == "--epochs" && i + 1 < argc) {
      cfg.epochs = std::stoul(argv[++i]);
    } else if (arg == "--batch" && i + 1 < argc) {
      cfg.batch_size = std::stoul(argv[++i]);
    } else if (arg == "--lr" && i + 1 < argc) {
      cfg.lr = std::stod(argv[++i]);
    } else if (arg == "--output" && i + 1 < argc) {
      cfg.output_model = argv[++i];
    } else if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      return 0;
    }
  }

  auto arch = mlp_arch(INPUT, cfg.hidden_dim, OUTPUT);
  auto field_fn = make_mlp_forward();
  auto params = init_params(arch);

  std::vector<Point> target_data;
  switch (cfg.manifold) {
  case ManifoldType::Euclidean:
    target_data = generate_target_euclidean(5000);
    break;
  case ManifoldType::Sphere:
    target_data = generate_target_sphere(5000);
    break;
  case ManifoldType::Torus:
    target_data = generate_target_torus(5000);
    break;
  }
  std::cout << "Generated " << target_data.size() << " target samples\n";

  std::mt19937 rng(42);
  std::normal_distribution<Scalar> base_dist(0, 1);

  // Only generate this for the Euclidean case. For Sphere and Torus we need
  // different metric classes, which requires separate template instantiation.
  EuclideanMetric<Traits> euclidean_metric;
  SphereMetric<Traits> sphere_metric;
  TorusMetric<Traits> torus_metric;

  // Dispatch by manifold type
  auto train = [&](auto& metric) {
    using Metric = std::decay_t<decltype(metric)>;
    ParametrizedVectorField<Traits, Metric> field(metric, field_fn);
    FlowIntegrator<Traits, Metric, decltype(field)> integrator(metric, field);
    AdjointSolver<Traits, Metric, decltype(field)> adjoint(metric, field);

    std::vector<Scalar> running_loss;
    Scalar best_loss = 1e10;

    for (size_t epoch = 0; epoch < cfg.epochs; ++epoch) {
      field.set_params(params);
      Scalar total_loss = 0;

      for (size_t b = 0; b < cfg.batch_size; ++b) {
        Point z{};
        for (size_t i = 0; i < N; ++i)
          z[i] = base_dist(rng);

        auto result = integrator.integrate(z, cfg.t0, cfg.t1, cfg.dt);
        const Point& xT = result.x_final;

        Scalar log_base = Scalar(0);
        for (size_t i = 0; i < N; ++i)
          log_base += -Scalar(0.5) * z[i] * z[i] - Scalar(0.5) * std::log(2 * M_PI);

        Scalar log_model = log_base - result.divergence_integral;
        Scalar log_target = log_target_density(xT, cfg.manifold);
        Scalar loss = -log_model + log_target;

        total_loss += loss;

        Tangent grad_xT = grad_log_target(xT, cfg.manifold);
        Cotangent aT{};
        for (size_t i = 0; i < N; ++i)
          aT.components[i] = -grad_xT.components[i];

        auto grad = adjoint.compute_gradient(z, cfg.t0, cfg.t1, cfg.dt, aT);
        for (size_t i = 0; i < params.size(); ++i)
          params[i] -= cfg.lr * grad[i] / Scalar(cfg.batch_size);
      }

      Scalar avg_loss = total_loss / Scalar(cfg.batch_size);
      running_loss.push_back(avg_loss);

      if ((epoch + 1) % 20 == 0) {
        std::cout << "Epoch " << (epoch + 1) << "/" << cfg.epochs << " loss=" << avg_loss
                  << std::endl;
      }

      if (avg_loss < best_loss) {
        best_loss = avg_loss;
        std::vector<Scalar> saved = params;
        save_params(cfg.output_model, saved, cfg.manifold);
      }
    }
    std::cout << "Best loss: " << best_loss << "\n";
    std::cout << "Model saved to " << cfg.output_model << "\n";
  };

  switch (cfg.manifold) {
  case ManifoldType::Euclidean:
    train(euclidean_metric);
    break;
  case ManifoldType::Sphere:
    train(sphere_metric);
    break;
  case ManifoldType::Torus:
    train(torus_metric);
    break;
  }

  return 0;
}
