cmake_minimum_required(VERSION 3.16)
project(NeutrinoDetSim VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#----------------------------------------------------------------------------
# Find Geant4 package (with UI and Vis drivers)
#
find_package(Geant4 REQUIRED ui_all vis_all)
include(${Geant4_USE_FILE})

#----------------------------------------------------------------------------
# Find ROOT package
#
find_package(ROOT REQUIRED COMPONENTS Core Hist Tree RIO)
include(${ROOT_USE_FILE})

#----------------------------------------------------------------------------
# Locate headers
#
include_directories(${PROJECT_SOURCE_DIR}/include)

#----------------------------------------------------------------------------
# Source files
#
file(GLOB SOURCES ${PROJECT_SOURCE_DIR}/src/*.cc)

#----------------------------------------------------------------------------
# Executable
#
add_executable(NeutrinoDetSim NeutrinoDetSim.cc ${SOURCES})

target_link_libraries(NeutrinoDetSim
  ${Geant4_LIBRARIES}
  ROOT::Core ROOT::Hist ROOT::Tree ROOT::RIO
)

#----------------------------------------------------------------------------
# Copy macros to build directory
#
file(GLOB MACROS ${PROJECT_SOURCE_DIR}/macros/*.mac)
foreach(_macro ${MACROS})
  configure_file(${_macro} ${PROJECT_BINARY_DIR}/macros/ COPYONLY)
endforeach()

file(GLOB ANALYSIS_SCRIPTS ${PROJECT_SOURCE_DIR}/analysis/*.C)
foreach(_script ${ANALYSIS_SCRIPTS})
  configure_file(${_script} ${PROJECT_BINARY_DIR}/analysis/ COPYONLY)
endforeach()

file(GLOB PLOT_SCRIPTS ${PROJECT_SOURCE_DIR}/plots/*.py)
foreach(_script ${PLOT_SCRIPTS})
  configure_file(${_script} ${PROJECT_BINARY_DIR}/plots/ COPYONLY)
endforeach()

#----------------------------------------------------------------------------
# Install
#
install(TARGETS NeutrinoDetSim DESTINATION bin)
