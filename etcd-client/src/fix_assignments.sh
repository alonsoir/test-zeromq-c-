#!/bin/bash

# Línea 283: set() - usar valor sin procesar
sed -i '283s/.*/        std::string processed_value = value; \/\/ DISABLED encryption/' etcd_client.cpp

# Línea 347: get() - usar valor sin procesar  
sed -i '347s/.*/        std::string processed_value = value; \/\/ DISABLED decryption/' etcd_client.cpp

# Línea 596: put_config() - usar json sin procesar
sed -i '596s/.*/        std::string processed_config = json_config; \/\/ DISABLED encryption/' etcd_client.cpp

echo "✅ Asignaciones arregladas"
