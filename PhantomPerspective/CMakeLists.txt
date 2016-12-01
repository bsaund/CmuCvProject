cmake_minimum_required(VERSION 2.8)
project( PhantomPerspective )
find_package( OpenCV REQUIRED )

add_library(stereo_match_initializer stereo_match_initializer.cpp)
target_link_libraries(stereo_match_initializer ${OpenCV_LIBS} )

add_library(perspective_matcher perspective_matcher.cpp)
target_link_libraries(perspective_matcher ${OpenCV_LIBS} )

add_executable(stereo_match_single stereo_match_single.cpp)
target_link_libraries( stereo_match_single
  ${OpenCV_LIBS} 
  stereo_match_initializer
  perspective_matcher)



# Commented because no main because visual studio
# add_executable(stereo_match_online stereo_match_online.cpp)
# target_link_libraries( stereo_match_online
#   ${OpenCV_LIBS} 
#   stereo_match_initializer
#   perspective_matcher)
