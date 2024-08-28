find_package(SDL2 REQUIRED)

# Include SDL2 include directories
include_directories(${SDL2_INCLUDE_DIRS})

add_executable(decoder_main decoder_main.cc)
# add_executable(testies_main testies_main.cc)

# Link the decoder library
target_link_libraries(decoder_main PUBLIC decoder)
# target_link_libraries(testies_main PUBLIC testies)


if(IPEX)
  target_link_libraries(decoder_main PUBLIC "${TORCH_IPEX_LIBRARIES}")
endif()

add_executable(label_checker_main label_checker_main.cc)
target_link_libraries(label_checker_main PUBLIC decoder)

if(TORCH)
 add_executable(api_main api_main.cc)
 target_link_libraries(api_main PUBLIC wenet_api)
endif()

if(WEBSOCKET)
  add_executable(websocket_server_main websocket_server_main.cc)
  target_link_libraries(websocket_server_main PUBLIC websocket)
endif()

if(GRPC)
  add_executable(grpc_server_main grpc_server_main.cc)
  target_link_libraries(grpc_server_main PUBLIC wenet_grpc)
  add_executable(grpc_client_main grpc_client_main.cc)
  target_link_libraries(grpc_client_main PUBLIC wenet_grpc)
endif()

if(HTTP)
  add_executable(http_client_main http_client_main.cc)
  target_link_libraries(http_client_main PUBLIC http)
  add_executable(http_server_main http_server_main.cc)
  target_link_libraries(http_server_main PUBLIC http)
endif()
