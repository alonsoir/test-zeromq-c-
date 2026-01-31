# CMake generated Testfile for 
# Source directory: /vagrant/rag-ingester
# Build directory: /vagrant/rag-ingester/build-tsan
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[FileWatcherTest]=] "/vagrant/rag-ingester/build-tsan/test_file_watcher")
set_tests_properties([=[FileWatcherTest]=] PROPERTIES  _BACKTRACE_TRIPLES "/vagrant/rag-ingester/CMakeLists.txt;178;add_test;/vagrant/rag-ingester/CMakeLists.txt;0;")
add_test([=[EventLoaderTest]=] "test_event_loader")
set_tests_properties([=[EventLoaderTest]=] PROPERTIES  _BACKTRACE_TRIPLES "/vagrant/rag-ingester/CMakeLists.txt;191;add_test;/vagrant/rag-ingester/CMakeLists.txt;0;")
