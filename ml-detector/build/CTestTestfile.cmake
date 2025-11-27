# CMake generated Testfile for 
# Source directory: /vagrant/ml-detector
# Build directory: /vagrant/ml-detector/build
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[test_classifier]=] "/vagrant/ml-detector/build/test_classifier")
set_tests_properties([=[test_classifier]=] PROPERTIES  _BACKTRACE_TRIPLES "/vagrant/ml-detector/CMakeLists.txt;411;add_test;/vagrant/ml-detector/CMakeLists.txt;0;")
add_test([=[test_feature_extractor]=] "/vagrant/ml-detector/build/test_feature_extractor")
set_tests_properties([=[test_feature_extractor]=] PROPERTIES  _BACKTRACE_TRIPLES "/vagrant/ml-detector/CMakeLists.txt;411;add_test;/vagrant/ml-detector/CMakeLists.txt;0;")
add_test([=[test_model_loader]=] "/vagrant/ml-detector/build/test_model_loader")
set_tests_properties([=[test_model_loader]=] PROPERTIES  _BACKTRACE_TRIPLES "/vagrant/ml-detector/CMakeLists.txt;411;add_test;/vagrant/ml-detector/CMakeLists.txt;0;")
add_test([=[test_detectors]=] "/vagrant/ml-detector/build/test_detectors")
set_tests_properties([=[test_detectors]=] PROPERTIES  _BACKTRACE_TRIPLES "/vagrant/ml-detector/CMakeLists.txt;439;add_test;/vagrant/ml-detector/CMakeLists.txt;0;")
add_test([=[RansomwareDetectorUnit]=] "/vagrant/ml-detector/build/test_ransomware_detector_unit")
set_tests_properties([=[RansomwareDetectorUnit]=] PROPERTIES  _BACKTRACE_TRIPLES "/vagrant/ml-detector/CMakeLists.txt;486;add_test;/vagrant/ml-detector/CMakeLists.txt;0;")
add_test([=[DetectorsUnit]=] "/vagrant/ml-detector/build/test_detectors_unit")
set_tests_properties([=[DetectorsUnit]=] PROPERTIES  _BACKTRACE_TRIPLES "/vagrant/ml-detector/CMakeLists.txt;505;add_test;/vagrant/ml-detector/CMakeLists.txt;0;")
