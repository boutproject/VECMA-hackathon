cmake_minimum_required(VERSION 3.14)

project(BOUT++-VECMA-hackathon
  DESCRIPTION "BOUT++ model exampels for the VECMA hackathon"
  LANGUAGES CXX)

# Don't generate the fieldops
set(BOUT_GENERATE_FIELDOPS OFF CACHE BOOL "" FORCE)
add_subdirectory(externals/BOUT++)

add_subdirectory(models/blob2d)
add_subdirectory(models/conduction)
