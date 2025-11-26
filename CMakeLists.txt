cmake_minimum_required(VERSION 3.20)
project(cyecca)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

include_directories(include)

ament_python_install_package(cyecca)

install(PROGRAMS
  scripts/rdd2_sim.py
  scripts/fixedwing_sim.py
  DESTINATION lib/${PROJECT_NAME};
  )

install(DIRECTORY
  launch config
  DESTINATION share/${PROJECT_NAME};
  )


if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  
  # Python tests
  # Run ./tools/format.sh to auto-format code with black + isort
  ament_add_pytest_test(${PROJECT_NAME}_pytest test
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )
endif()

ament_package()

# vi: ts=2 sw=2 et
