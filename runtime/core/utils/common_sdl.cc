#include "utils/common_sdl.h"

audio_async::audio_async(int len_ms) {
  m_len_ms = len_ms;
  m_running = false;
}

audio_async::~audio_async() {
  if (m_dev_id_in) {
    SDL_CloseAudioDevice(m_dev_id_in);
  }
}

bool audio_async::init(int capture_id, int sample_rate) {
  SDL_LogSetPriority(SDL_LOG_CATEGORY_APPLICATION, SDL_LOG_PRIORITY_INFO);

  if (SDL_Init(SDL_INIT_AUDIO) < 0) {
    SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "Couldn't initialize SDL: %s\n",
                 SDL_GetError());
    return false;
  }

  SDL_SetHintWithPriority(SDL_HINT_AUDIO_RESAMPLING_MODE, "high",
                          SDL_HINT_OVERRIDE);

  int nDevices = SDL_GetNumAudioDevices(SDL_TRUE);
  for (int i = 0; i < nDevices; i++) {
  }

  SDL_AudioSpec capture_spec_requested, capture_spec_obtained;
  SDL_zero(capture_spec_requested);
  SDL_zero(capture_spec_obtained);

  capture_spec_requested.freq = sample_rate;
  capture_spec_requested.format = AUDIO_S16SYS;
  capture_spec_requested.channels = 1;
  capture_spec_requested.samples = 1024;
  capture_spec_requested.callback = [](void* userdata, uint8_t* stream,
                                       int len) {
    audio_async* audio = (audio_async*)userdata;
    audio->callback(stream, len);
  };
  capture_spec_requested.userdata = this;

  if (capture_id >= 0) {
    fprintf(stderr, "%s: attempt to open capture device %d : '%s' ...\n",
            __func__, capture_id, SDL_GetAudioDeviceName(capture_id, SDL_TRUE));
    m_dev_id_in = SDL_OpenAudioDevice(
        SDL_GetAudioDeviceName(capture_id, SDL_TRUE), SDL_TRUE,
        &capture_spec_requested, &capture_spec_obtained, 0);
  } else {
    fprintf(stderr, "%s: attempt to open default capture device ...\n",
            __func__);
    m_dev_id_in = SDL_OpenAudioDevice(
        nullptr, SDL_TRUE, &capture_spec_requested, &capture_spec_obtained, 0);
  }

  if (!m_dev_id_in) {
    fprintf(stderr, "%s: couldn't open an audio device for capture: %s!\n",
            __func__, SDL_GetError());
    m_dev_id_in = 0;

    return false;
  } else {
    fprintf(stderr, "%s: obtained spec for input device (SDL Id = %d):\n",
            __func__, m_dev_id_in);
    fprintf(stderr, "%s:     - sample rate:       %d\n", __func__,
            capture_spec_obtained.freq);
    fprintf(stderr, "%s:     - format:            %d (required: %d)\n",
            __func__, capture_spec_obtained.format,
            capture_spec_requested.format);
    fprintf(stderr, "%s:     - channels:          %d (required: %d)\n",
            __func__, capture_spec_obtained.channels,
            capture_spec_requested.channels);
    fprintf(stderr, "%s:     - samples per frame: %d\n", __func__,
            capture_spec_obtained.samples);
  }

  m_sample_rate = capture_spec_obtained.freq;
  m_audio.clear();

  return true;
}

bool audio_async::resume() {
  if (!m_dev_id_in || m_running) {
    return false;
  }

  SDL_PauseAudioDevice(m_dev_id_in, 0);
  m_running = true;

  return true;
}

bool audio_async::pause() {
  if (!m_dev_id_in || !m_running) {
    return false;
  }

  SDL_PauseAudioDevice(m_dev_id_in, 1);
  m_running = false;

  return true;
}

bool audio_async::clear() {
  if (!m_dev_id_in || !m_running) {
    return false;
  }

  std::lock_guard<std::mutex> lock(m_mutex);
  m_audio.clear();

  return true;
}

void audio_async::callback(uint8_t* stream, int len) {
  if (!m_running) {
    return;
  }

  size_t n_samples = len / sizeof(int16_t);
  std::lock_guard<std::mutex> lock(m_mutex);
  for (size_t i = 0; i < n_samples; i++) {
    m_audio.push_back(((int16_t*)stream)[i]);
  }
}

void audio_async::get(std::vector<int16_t>& result) {
  if (!m_dev_id_in || !m_running) {
    return;
  }

  std::lock_guard<std::mutex> lock(m_mutex);
  result = std::move(m_audio);
  m_audio.clear();
}

bool sdl_poll_events() {
  SDL_Event event;
  while (SDL_PollEvent(&event)) {
    if (event.type == SDL_QUIT) {
      return false;
    }
  }

  return true;
}
