# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/vagrant/firewall-acl-agent/build/_deps/googletest-src"
  "/vagrant/firewall-acl-agent/build/_deps/googletest-build"
  "/vagrant/firewall-acl-agent/build/_deps/googletest-subbuild/googletest-populate-prefix"
  "/vagrant/firewall-acl-agent/build/_deps/googletest-subbuild/googletest-populate-prefix/tmp"
  "/vagrant/firewall-acl-agent/build/_deps/googletest-subbuild/googletest-populate-prefix/src/googletest-populate-stamp"
  "/vagrant/firewall-acl-agent/build/_deps/googletest-subbuild/googletest-populate-prefix/src"
  "/vagrant/firewall-acl-agent/build/_deps/googletest-subbuild/googletest-populate-prefix/src/googletest-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/vagrant/firewall-acl-agent/build/_deps/googletest-subbuild/googletest-populate-prefix/src/googletest-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/vagrant/firewall-acl-agent/build/_deps/googletest-subbuild/googletest-populate-prefix/src/googletest-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
