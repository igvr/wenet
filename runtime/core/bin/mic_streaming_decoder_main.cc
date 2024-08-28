#include <SDL.h>
#include <chrono>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "decoder/params.h"
#include "utils/common_sdl.h"
#include "utils/flags.h"
#include "utils/string.h"
#include "utils/timer.h"
#include "utils/utils.h"

DEFINE_int32(device, -1, "audio device id");
DEFINE_string(vad_model_path, "silero_vad.onnx", "silero vad model");
DEFINE_bool(save_audio, false, "save audio to wav file");

std::shared_ptr<wenet::DecodeOptions> g_decode_config;
std::shared_ptr<wenet::FeaturePipelineConfig> g_feature_config;
std::shared_ptr<wenet::DecodeResource> g_decode_resource;

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  google::InitGoogleLogging(argv[0]);

  // g_decode_config = wenet::InitDecodeOptionsFromFlags();
  // g_feature_config = wenet::InitFeaturePipelineConfigFromFlags();
  // g_decode_resource = wenet::InitDecodeResourceFromFlags();

  // feature_pipeline =
  //       std::make_shared<wenet::FeaturePipeline>(*g_feature_config);
  // decoder = std::make_shared<AsrDecoder>(feature_pipeline_, decode_resource_,
  //                                         *decode_config_);
  // decode_thread =
  //     std::make_shared<std::thread>(&DecodeThreadFunc, this);

  wav_writer wavWriter;

  // Save WAV file if required
  if (FLAGS_save_audio) {
    time_t now = time(0);
    char buffer[80];
    strftime(buffer, sizeof(buffer), "%Y%m%d%H%M%S", localtime(&now));
    std::string filename = std::string(buffer) + ".wav";

    wavWriter.open(filename, FLAGS_sample_rate, 16, 1);
  }

  audio_async audio(20000);  // Buffer duration in milliseconds
  if (!audio.init(FLAGS_device, FLAGS_sample_rate)) {
    std::cerr << "Failed to initialize audio!" << std::endl;
    SDL_Quit();
    return 1;
  }
  audio.resume();

  bool is_running = true;
  std::vector<int16_t> audio_data;

  while (is_running) {
    is_running = sdl_poll_events();

    // Retrieve all available audio data since the last call to get
    audio.get(audio_data);

    if (!audio_data.empty() && FLAGS_save_audio) {
      wavWriter.write(audio_data.data(), audio_data.size());
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  audio.pause();
  SDL_Quit();

  return 0;
}