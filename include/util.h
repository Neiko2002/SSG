#ifndef EFANNA2E_UTIL_H
#define EFANNA2E_UTIL_H

#include <random>

namespace efanna2e {

    #ifdef __AVX__
      #define DATA_ALIGN_FACTOR 8
    #else
    #ifdef __SSE2__
      #define DATA_ALIGN_FACTOR 4
    #else
      #define DATA_ALIGN_FACTOR 1
    #endif
    #endif

    void GenRandom(std::mt19937& rng, unsigned* addr, unsigned size, unsigned N);

    float* load_data(const char* filename, unsigned& num, unsigned& dim);

    float* data_align(float* data_ori, unsigned point_num, unsigned& dim);

}  // namespace efanna2e

#endif  // EFANNA2E_UTIL_H
