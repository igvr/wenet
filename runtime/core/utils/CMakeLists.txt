find_package(SDL2 REQUIRED)

# Include SDL2 include directories
include_directories(${SDL2_INCLUDE_DIRS})

# Add the utils library
add_library(utils STATIC
  string.cc
  utils.cc
  common_sdl.cc
  wav_writer.cc
)

if(NOT ANDROID)
  if(MSVC)
    target_link_libraries(utils PUBLIC fst ${SDL2_LIBRARIES})
  else()
    target_link_libraries(utils PUBLIC fst dl ${SDL2_LIBRARIES})
  endif()
endif()