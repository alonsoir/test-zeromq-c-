# CMake generated Testfile for 
# Source directory: /vagrant/ml-detector
# Build directory: /vagrant/ml-detector/build
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[RansomwareDetectorUnit]=] "/vagrant/ml-detector/build/test_ransomware_detector_unit")
set_tests_properties([=[RansomwareDetectorUnit]=] PROPERTIES  _BACKTRACE_TRIPLES "/vagrant/ml-detector/CMakeLists.txt;505;add_test;/vagrant/ml-detector/CMakeLists.txt;0;")
add_test([=[DetectorsUnit]=] "/vagrant/ml-detector/build/test_detectors_unit")
set_tests_properties([=[DetectorsUnit]=] PROPERTIES  _BACKTRACE_TRIPLES "/vagrant/ml-detector/CMakeLists.txt;527;add_test;/vagrant/ml-detector/CMakeLists.txt;0;")
