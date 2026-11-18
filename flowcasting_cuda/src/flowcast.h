#pragma once
#include <cstdint>

#ifdef _WIN32
#define FC_API __declspec(dllexport)
#else
#define FC_API __attribute__((visibility("default")))
#endif

// Keep in sync with flowcast/optical_flow_cuda.py. The ABI is deliberately C-only.
struct FCConfig {
    int pairs, gap_count, gaps[8];
    int align_motion, confidence_weighting, smooth_motion, growth_decay, multiscale;
    int midpoint, zero_boundary, scale_count;
    float recency_decay, smoothing_sigma, photometric_scale, consistency_scale;
    float growth_sigma, growth_cap, growth_lifetime, sigmas[8], lifetimes[8];
};
struct FCTiming { float upload_ms, motion_ms, evolution_ms, download_ms; };

extern "C" {
FC_API void* fc_create(int device);
FC_API void fc_destroy(void* context);
FC_API const char* fc_error();
FC_API const char* fc_device(void* context);
FC_API int fc_runtime_version();
// supplied_flows: optional CPU DIS fields, (gap, age, direction, H, W, 2).
// Each gap uses min(pairs, T-gap) ages; backward then forward for every age.
FC_API int fc_predict(void* context, const uint8_t* frames, int t, int h, int w,
                      int steps, const FCConfig* config, const float* supplied_flows,
                      uint8_t* prediction, float* motion, FCTiming* timing);
// Component check for known fields, using the same kernels as the full predictor.
FC_API int fc_advect(void* context, const float* last, const float* motion,
                     const float* trend, int h, int w, int steps,
                     const FCConfig* config, uint8_t* prediction);
}
