# CMake generated Testfile for 
# Source directory: /vagrant/crypto-transport/tests
# Build directory: /vagrant/crypto-transport/build/tests
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(test_crypto "/vagrant/crypto-transport/build/tests/test_crypto")
set_tests_properties(test_crypto PROPERTIES  _BACKTRACE_TRIPLES "/vagrant/crypto-transport/tests/CMakeLists.txt;28;add_test;/vagrant/crypto-transport/tests/CMakeLists.txt;32;add_crypto_transport_test;/vagrant/crypto-transport/tests/CMakeLists.txt;0;")
add_test(test_compression "/vagrant/crypto-transport/build/tests/test_compression")
set_tests_properties(test_compression PROPERTIES  _BACKTRACE_TRIPLES "/vagrant/crypto-transport/tests/CMakeLists.txt;28;add_test;/vagrant/crypto-transport/tests/CMakeLists.txt;33;add_crypto_transport_test;/vagrant/crypto-transport/tests/CMakeLists.txt;0;")
add_test(test_integration "/vagrant/crypto-transport/build/tests/test_integration")
set_tests_properties(test_integration PROPERTIES  _BACKTRACE_TRIPLES "/vagrant/crypto-transport/tests/CMakeLists.txt;28;add_test;/vagrant/crypto-transport/tests/CMakeLists.txt;34;add_crypto_transport_test;/vagrant/crypto-transport/tests/CMakeLists.txt;0;")
