//
// Created by 付聪 on 2017/6/21.
//

#include "index_random.h"
#include "index_ssg.h"
#include "util.h"


#ifdef _OPENMP
#include <omp.h>
#endif

static void load_data(const char* filename, float*& data, unsigned& num, unsigned& dim) {  // load data with sift10K pattern
  std::ifstream in(filename, std::ios::binary);
  if (!in.is_open()) {
    std::cout << "open file error" << std::endl;
    exit(-1);
  }
  in.read((char*)&dim, 4);
  in.seekg(0, std::ios::end);
  std::ios::pos_type ss = in.tellg();
  size_t fsize = (size_t)ss;
  num = (unsigned)(fsize / (dim + 1) / 4);
  data = new float[(size_t)num * (size_t)dim];

  in.seekg(0, std::ios::beg);
  for (size_t i = 0; i < num; i++) {
    in.seekg(4, std::ios::cur);
    in.read((char*)(data + i * dim), dim * 4);
  }
  in.close();
}

int main(int argc, char** argv) {

  std::cout << "DATA_ALIGN_FACTOR " << DATA_ALIGN_FACTOR << std::endl;
  #ifdef _OPENMP
        omp_set_dynamic(0);     // Explicitly disable dynamic teams
        omp_set_num_threads(1); // Use 1 threads for all consecutive parallel regions

        std::cout << "_OPENMP " << omp_get_num_threads() << " threads" << std::endl;
  #endif

  #ifdef __AVX__
    std::cout << "__AVX__ is set" << std::endl;
  #endif

  auto object_file      = R"(c:/Data/Feature/SIFT1M/SIFT1M/sift_base.fvecs)";
  auto ssg_file         = R"(c:/Data/Feature/SIFT1M/ssg/sift.ssg)";
  auto efanna_file      = std::string("c:/Data/Feature/SIFT1M/efanna/sift_200nn.graph");

  // load feature vectors
  float* data_load = NULL;
  unsigned points_num, dim;
  load_data(object_file, data_load, points_num, dim);
  data_load = efanna2e::data_align(data_load, points_num, dim); // align the data before build

  // https://github.com/Neiko2002/SSG
  unsigned L = 100;
  unsigned R = 50;
  unsigned A = 60;
  unsigned seed = 7;
  srand(seed);

  // create the index
  efanna2e::IndexSSG index(dim, points_num, efanna2e::L2, nullptr);

  efanna2e::Parameters paras;
  paras.Set<unsigned>("L", L);
  paras.Set<unsigned>("R", R);
  paras.Set<unsigned>("A", A);
  paras.Set<unsigned>("n_try", 10);
  paras.Set<std::string>("nn_graph_path", efanna_file);

  auto s = std::chrono::high_resolution_clock::now();
  index.Build(points_num, data_load, paras);
  auto e = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = e - s;

  std::cout << "indexing time: " << diff.count() << "\n";
  index.Save(ssg_file);

  return 0;
}