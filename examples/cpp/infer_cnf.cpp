#include <cmath>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

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
using Params = std::vector<Scalar>;

enum class ManifoldType { Euclidean, Sphere, Torus };

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

bool load_params(const std::string& path, std::vector<Scalar>& params, ManifoldType& mt) {
  std::ifstream f(path, std::ios::binary);
  if (!f)
    return false;
  int mt_int;
  f.read(reinterpret_cast<char*>(&mt_int), sizeof(int));
  mt = static_cast<ManifoldType>(mt_int);
  size_t sz;
  f.read(reinterpret_cast<char*>(&sz), sizeof(size_t));
  params.resize(sz);
  f.read(reinterpret_cast<char*>(params.data()), sz * sizeof(Scalar));
  return true;
}

void write_frame(const std::string& dir, size_t frame_idx, const std::vector<Point>& points,
                 ManifoldType mt) {
  std::ostringstream ss;
  ss << dir << "/frame_" << std::setw(4) << std::setfill('0') << frame_idx << ".csv";
  std::ofstream f(ss.str());
  f << "x,y,z\n";
  for (const auto& p : points) {
    std::array<Scalar, 3> xyz;
    switch (mt) {
    case ManifoldType::Sphere:
      xyz = sphere_to_cartesian(p);
      break;
    case ManifoldType::Torus:
      xyz = torus_to_cartesian(p);
      break;
    default:
      xyz = {p[0], p[1], Scalar(0)};
      break;
    }
    f << xyz[0] << "," << xyz[1] << "," << xyz[2] << "\n";
  }
}

void write_trajectories_csv(
    const std::string& path, size_t n_traj_points,
    const std::vector<std::pair<Scalar, std::vector<std::array<Scalar, 3>>>>& data) {
  std::ofstream f(path);
  // header: t, x0, y0, z0, x1, y1, z1, ...
  f << "t";
  for (size_t k = 0; k < n_traj_points; ++k)
    f << ",x" << k << ",y" << k << ",z" << k;
  f << "\n";
  for (const auto& row : data) {
    f << row.first;
    for (size_t k = 0; k < n_traj_points; ++k) {
      const auto& xyz = row.second[k];
      f << "," << xyz[0] << "," << xyz[1] << "," << xyz[2];
    }
    f << "\n";
  }
  std::cout << "Wrote " << data.size() << " trajectory steps to " << path << "\n";
}

void print_usage(const char* prog) {
  std::cerr << "Usage: " << prog
            << " --model <path> --output-dir <dir> [--n-points 1000] [--n-frames 50]\n"
            << "               [--trajectories N] [--trajectory-steps M]\n";
}

int main(int argc, char** argv) {
  std::string model_path;
  std::string output_dir = "inference_frames";
  size_t n_points = 1000;
  size_t n_frames = 50;
  size_t n_trajectories = 0;
  size_t trajectory_steps = 100;
  Scalar dt = 0.05;
  Scalar t0 = 0;
  Scalar t1 = 1;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--model" && i + 1 < argc)
      model_path = argv[++i];
    else if (arg == "--output-dir" && i + 1 < argc)
      output_dir = argv[++i];
    else if (arg == "--n-points" && i + 1 < argc)
      n_points = std::stoul(argv[++i]);
    else if (arg == "--n-frames" && i + 1 < argc)
      n_frames = std::stoul(argv[++i]);
    else if (arg == "--trajectories" && i + 1 < argc)
      n_trajectories = std::stoul(argv[++i]);
    else if (arg == "--trajectory-steps" && i + 1 < argc)
      trajectory_steps = std::stoul(argv[++i]);
    else if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      return 0;
    }
  }

  if (model_path.empty()) {
    std::cerr << "Error: --model is required\n";
    print_usage(argv[0]);
    return 1;
  }

  ManifoldType mt;
  std::vector<Scalar> params;
  if (!load_params(model_path, params, mt)) {
    std::cerr << "Failed to load model from " << model_path << "\n";
    return 1;
  }
  std::cout << "Loaded model with " << params.size() << " parameters, manifold type "
            << static_cast<int>(mt) << "\n";

  std::filesystem::create_directories(output_dir);

  auto field_fn = make_mlp_forward();

  std::mt19937 rng(42);
  std::normal_distribution<Scalar> base_dist(0, 1);

  std::vector<Point> initial_points;
  initial_points.reserve(n_points);
  for (size_t i = 0; i < n_points; ++i) {
    Point z{};
    for (size_t j = 0; j < N; ++j)
      z[j] = base_dist(rng);
    initial_points.push_back(z);
  }

  auto run_inference = [&](auto& metric) {
    using Metric = std::decay_t<decltype(metric)>;
    ParametrizedVectorField<Traits, Metric> field(metric, field_fn);
    field.set_params(params);
    FlowIntegrator<Traits, Metric, decltype(field)> integrator(metric, field);

    write_frame(output_dir, 0, initial_points, mt);

    Scalar dt_frame = (t1 - t0) / Scalar(n_frames);

    for (size_t frame = 1; frame <= n_frames; ++frame) {
      Scalar t_frame = t0 + Scalar(frame) * dt_frame;

      std::vector<Point> frame_points;
      frame_points.reserve(initial_points.size());
      for (const auto& z : initial_points) {
        auto result = integrator.integrate(z, t0, t_frame, dt, false);
        frame_points.push_back(result.x_final);
      }
      write_frame(output_dir, frame, frame_points, mt);
    }

    std::cout << "Wrote " << (n_frames + 1) << " frames to " << output_dir << "\n";

    // ---- trajectory export -------------------------------------------
    if (n_trajectories > 0) {
      size_t actual_n = std::min(n_trajectories, n_points);
      Scalar dt_traj = (t1 - t0) / Scalar(trajectory_steps);

      std::vector<std::vector<std::array<Scalar, 3>>> per_point_trajs;
      size_t max_steps = 0;

      for (size_t k = 0; k < actual_n; ++k) {
        auto result = integrator.integrate(initial_points[k], t0, t1, dt_traj, true);
        std::vector<std::array<Scalar, 3>> cart_traj;
        cart_traj.reserve(result.trajectory.size());
        for (const auto& entry : result.trajectory) {
          const auto& pt = entry.state;
          std::array<Scalar, 3> xyz;
          switch (mt) {
          case ManifoldType::Sphere:
            xyz = sphere_to_cartesian(pt);
            break;
          case ManifoldType::Torus:
            xyz = torus_to_cartesian(pt);
            break;
          default:
            xyz = {pt[0], pt[1], Scalar(0)};
            break;
          }
          cart_traj.push_back(xyz);
        }
        per_point_trajs.push_back(std::move(cart_traj));
        max_steps = std::max(max_steps, per_point_trajs.back().size());
      }

      std::vector<std::pair<Scalar, std::vector<std::array<Scalar, 3>>>> per_step_data;
      per_step_data.reserve(max_steps);
      for (size_t s = 0; s < max_steps; ++s) {
        Scalar t = t0 + Scalar(s) * dt_traj;
        if (t > t1)
          t = t1;
        std::vector<std::array<Scalar, 3>> step_xyzs;
        step_xyzs.reserve(actual_n);
        for (size_t k = 0; k < actual_n; ++k) {
          if (s < per_point_trajs[k].size())
            step_xyzs.push_back(per_point_trajs[k][s]);
          else
            step_xyzs.push_back(per_point_trajs[k].back());
        }
        per_step_data.emplace_back(t, std::move(step_xyzs));
      }

      write_trajectories_csv(output_dir + "/trajectories.csv", actual_n, per_step_data);
    }
  };

  EuclideanMetric<Traits> euclidean_metric;
  SphereMetric<Traits> sphere_metric;
  TorusMetric<Traits> torus_metric;

  switch (mt) {
  case ManifoldType::Euclidean:
    run_inference(euclidean_metric);
    break;
  case ManifoldType::Sphere:
    run_inference(sphere_metric);
    break;
  case ManifoldType::Torus:
    run_inference(torus_metric);
    break;
  }

  return 0;
}
