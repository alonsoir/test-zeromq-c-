#!/bin/bash
# Eliminar forward declarations problemáticas
sed -i '
/namespace compression {/,/^}/d
/namespace crypto {/,/^}/d
' etcd_client.cpp

# Comentar process_outgoing_data y process_incoming_data
sed -i 's/std::string process_outgoing_data/\/\/ DISABLED: std::string process_outgoing_data/' etcd_client.cpp
sed -i 's/std::string process_incoming_data/\/\/ DISABLED: std::string process_incoming_data/' etcd_client.cpp

# Comentar llamadas a estas funciones
sed -i 's/pImpl->process_outgoing_data/\/\/ pImpl->process_outgoing_data/' etcd_client.cpp
sed -i 's/pImpl->process_incoming_data/\/\/ pImpl->process_incoming_data/' etcd_client.cpp
