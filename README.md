```markdown
# POC C++ Messaging Lab

Este proyecto es una **Prueba de Concepto (POC)** que implementa un laboratorio de comunicación entre dos servicios en **C++** utilizando **ZeroMQ**.  

Actualmente el laboratorio permite intercambiar **mensajes de texto plano** entre:

- `service1`
- `service2`

El objetivo es verificar que los servicios pueden comunicarse correctamente en un entorno controlado antes de añadir soporte para **Protobuf compilado**.

---

## Estructura del Proyecto

```

.
├── service1/                # Código fuente del servicio 1
├── service2/                # Código fuente del servicio 2
├── Vagrantfile              # Máquina virtual Ubuntu Server
├── docker-compose.yml       # Orquestación de servicios con Docker
├── Makefile                 # Comandos de compilación y laboratorio
├── README.md
└── .gitignore

````

---

## Requisitos Previos

- **Vagrant** >= 2.x  
- **VirtualBox** u otro proveedor compatible  
- **C++ Compiler** (g++ >= 9)  
- **CMake** >= 3.18  
- **ZeroMQ** y su librería de C++ (`libzmq3-dev`)  

Opcional:

- Docker y Docker Compose si quieres levantar los servicios en contenedores automáticamente.

---

## Configuración del Entorno

1. Levantar la máquina virtual:

```bash
vagrant up
vagrant ssh
````

2. Instalar dependencias dentro de la VM:

```bash
sudo apt update
sudo apt install -y build-essential cmake libzmq3-dev
```

3. Clonar el proyecto (si no está ya dentro de la VM):

```bash
git clone <repo-url>
cd <repo-folder>
```

---

## Compilación de los Servicios

Desde la raíz del proyecto, usar `Makefile` para compilar en la maquina virtual:
Las librerias están optimizadas para linux. Esta poc ha sido desarrollada en osx.
No correrá en osx, tienes que levantar Vagrant para ver la demo funcionando.

```bash
make native-build
```

Esto generará:

* `bin/service1_exe`
* `bin/service2_exe`

---

## Ejecución del Laboratorio

Para levantar **el laboratorio completo con Docker sobre la VM**, usar:

```bash
make lab-start
```

Esto hará:

1. Levantar la VM con Vagrant (`vagrant up`)
2. Construir y ejecutar los contenedores Docker con `docker-compose`
3. Ejecutar `service1` y `service2` intercambiando mensajes de texto plano

### Ejecución local (sin Docker)

```bash
make native-run
```

Esto ejecuta `service1` en background y luego `service2`, mostrando la comunicación en la terminal.

---

## Limpieza

Para eliminar binarios y contenedores:

```bash
make clean
```

```bash
┌<▸> ~/C/test-zeromq-docker 
└➤ git push -u origin main --force                                                                                                                                                             

Enumerating objects: 22, done.
Counting objects: 100% (22/22), done.
Delta compression using up to 16 threads
Compressing objects: 100% (21/21), done.
Writing objects: 100% (22/22), 11.61 KiB | 3.87 MiB/s, done.
Total 22 (delta 2), reused 0 (delta 0), pack-reused 0
remote: Resolving deltas: 100% (2/2), done.
To https://github.com/alonsoir/test-zeromq-c-.git
 + 07f74b1...5046144 main -> main (forced update)
branch 'main' set up to track 'origin/main'.
┌<▸> ~/C/test-zeromq-docker 
└➤ make native-build                                                                                                                                                                           
🔨 Compilando service1... 
service1/main.cpp:1:10: fatal error: 'zmq.hpp' file not found
    1 | #include <zmq.hpp>
      |          ^~~~~~~~~
1 error generated.                                                                                                                                                                             
make: *** [bin/service1_exe] Error 1
┌<▪> ~/C/test-zeromq-docker 
└➤ make clean                                                                                                                                                                                  
🧹 Limpiando... 
Deleted Containers:
835536073cada836c0536488fde9b60e417c04c7a77ef510e9e6a25a81f11a20
e6a69fde4aa91c7b508fbf3ba63af9d09207813a338b0977a42596d3040b9881

Deleted Networks:
vagrant_zeromq-net

Deleted build cache objects:
tgnbvsisthjkcrgxx4f1ruwax
lrp8ijafmo8nw3u8gjpqh00ek
pqryex25kotsqrfgujiqwvbez
e5uz3ewrmryuc7ivv2hzmyyvx
34zz56t07s3w5l9gkj3kla63a
s28lsc6j5plqncg59kqx7430w
nowhjfxdnbkfkovtecfqo8gz8
at83agspt8o2l6dfqr9gxssla
sb4y3zh5imeoqnorfpvhgymmw
d8tvwqzg8c8secn51ujsd8abe
8y5jpbo9whu5h8y780de9hzgh
ubd41dc8uiry6oz7mmr1w5n4r
mz6dmrkwmbzz5j0j276qwfi32
c2jo39wlxzbo8p464dj4ak3sb
57huujtblrztb2x1kgf44qi6j
vyfyi7gc1vbxe51h9p1ovai9m
upxbug2gozuzr992exschsilm
yuj3ot1y76grraldsyz03n74c
9uusfjuuskydau7wnefg7gwtg
r4aya6o53d1ghphvvzi5g98wk
3b7qj90giop059e7ok5ji71ey
uj7cj01c343030081jdkq14qh
n6y8zmwvxvzes6lh3qscojdr5

Total reclaimed space: 927.6MB
┌<▸> ~/C/test-zeromq-docker 
└➤ make lab-start                                                                                                                                                                              
🚀 Iniciando laboratorio ZeroMQ... 
🖥️  Levantando VM... 
Bringing machine 'default' up with 'virtualbox' provider...
==> default: Checking if box 'ubuntu/focal64' version '20240821.0.1' is up to date...
==> default: Machine already provisioned. Run `vagrant provision` or use the `--provision`
==> default: flag to force provisioning. Provisioners marked to run always will still run.
⏳ Esperando que VM esté lista... 
🐳 Ejecutando POC en Docker... 
🚀 Construyendo contenedores...
[+] Building 1.4s (34/34) FINISHED                                                                                                                                              docker:default
 => [service1 internal] load build definition from Dockerfile.service1                                                                                                                    0.0s
 => => transferring dockerfile: 1.33kB                                                                                                                                                    0.0s 
 => [service2 internal] load metadata for docker.io/library/ubuntu:22.04                                                                                                                  1.1s 
 => [service1 internal] load .dockerignore                                                                                                                                                0.0s
 => => transferring context: 2B                                                                                                                                                           0.0s 
 => [service2 builder 1/8] FROM docker.io/library/ubuntu:22.04@sha256:4e0171b9275e12d375863f2b3ae9ce00a4c53ddda176bd55868df97ac6f21a6e                                                    0.0s 
 => [service1 internal] load build context                                                                                                                                                0.0s 
 => => transferring context: 704B                                                                                                                                                         0.0s
 => CACHED [service2 zeromq-build 2/9] RUN apt-get update && apt-get install -y     build-essential pkg-config git cmake libtool autoconf automake                                        0.0s 
 => CACHED [service2 zeromq-build 3/9] WORKDIR /tmp                                                                                                                                       0.0s 
 => CACHED [service2 zeromq-build 4/9] RUN git clone --depth 1 https://github.com/zeromq/libzmq.git                                                                                       0.0s 
 => CACHED [service2 zeromq-build 5/9] WORKDIR /tmp/libzmq                                                                                                                                0.0s 
 => CACHED [service2 zeromq-build 6/9] RUN mkdir build && cd build && cmake .. && make -j$(nproc) && make install                                                                         0.0s 
 => CACHED [service2 zeromq-build 7/9] WORKDIR /tmp                                                                                                                                       0.0s 
 => CACHED [service2 zeromq-build 8/9] RUN git clone --depth 1 https://github.com/zeromq/cppzmq.git                                                                                       0.0s 
 => CACHED [service2 zeromq-build 9/9] RUN cp cppzmq/zmq.hpp /usr/local/include/                                                                                                          0.0s 
 => CACHED [service2 stage-2 2/4] COPY --from=zeromq-build /usr/local/lib/ /usr/local/lib/                                                                                                0.0s 
 => CACHED [service2 stage-2 3/4] RUN ldconfig                                                                                                                                            0.0s 
 => CACHED [service2 builder 2/8] RUN apt-get update && apt-get install -y build-essential cmake pkg-config                                                                               0.0s 
 => CACHED [service2 builder 3/8] WORKDIR /app                                                                                                                                            0.0s 
 => CACHED [service1 builder 4/8] COPY service1/main.cpp ./main.cpp                                                                                                                       0.0s 
 => CACHED [service1 builder 5/8] COPY service1/main.h ./main.h                                                                                                                           0.0s 
 => CACHED [service1 builder 6/8] COPY --from=zeromq-build /usr/local/include/ /usr/local/include/                                                                                        0.0s 
 => CACHED [service1 builder 7/8] COPY --from=zeromq-build /usr/local/lib/ /usr/local/lib/                                                                                                0.0s 
 => CACHED [service1 builder 8/8] RUN g++ -std=c++20 main.cpp -lzmq -o service1_exe                                                                                                       0.0s 
 => CACHED [service1 stage-2 4/4] COPY --from=builder /app/service1_exe /usr/local/bin/service1_exe                                                                                       0.0s 
 => [service1] exporting to image                                                                                                                                                         0.0s 
 => => exporting layers                                                                                                                                                                   0.0s 
 => => writing image sha256:b4853d83ae87992ecd2319b42f34b2201b6a0cbde0508a1f9a8cb49bd9c05b95                                                                                              0.0s 
 => => naming to docker.io/library/vagrant-service1                                                                                                                                       0.0s 
 => [service2 internal] load build definition from Dockerfile.service2                                                                                                                    0.0s 
 => => transferring dockerfile: 1.33kB                                                                                                                                                    0.0s 
 => [service2 internal] load .dockerignore                                                                                                                                                0.0s 
 => => transferring context: 2B                                                                                                                                                           0.0s 
 => [service2 internal] load build context                                                                                                                                                0.0s 
 => => transferring context: 585B                                                                                                                                                         0.0s 
 => CACHED [service2 builder 4/8] COPY service2/main.cpp ./main.cpp                                                                                                                       0.0s 
 => CACHED [service2 builder 5/8] COPY service2/main.h ./main.h                                                                                                                           0.0s 
 => CACHED [service2 builder 6/8] COPY --from=zeromq-build /usr/local/include/ /usr/local/include/                                                                                        0.0s 
 => CACHED [service2 builder 7/8] COPY --from=zeromq-build /usr/local/lib/ /usr/local/lib/                                                                                                0.0s 
 => CACHED [service2 builder 8/8] RUN g++ -std=c++20 main.cpp -lzmq -o service2_exe                                                                                                       0.0s 
 => CACHED [service2 stage-2 4/4] COPY --from=builder /app/service2_exe /usr/local/bin/service2_exe                                                                                       0.0s 
 => [service2] exporting to image                                                                                                                                                         0.0s 
 => => exporting layers                                                                                                                                                                   0.0s 
 => => writing image sha256:f21a817e2dc84d738f548f5db5a668aaec319628a823ec46d9bbcbdd148b2913                                                                                              0.0s 
 => => naming to docker.io/library/vagrant-service2                                                                                                                                       0.0s 
📤 Levantando contenedores en background...                                                                                                                                                    
[+] Running 3/3
 ✔ Network vagrant_zeromq-net    Created                                                                                                                                                  0.1s 
 ✔ Container vagrant-service1-1  Started                                                                                                                                                  0.1s 
 ✔ Container vagrant-service2-1  Started                                                                                                                                                  0.0s 
⏳ Esperando 3 segundos para que service1 esté listo...
📌 Mostrando logs de service1 y service2...
vagrant-service2-1  |  Sent: Hello World from Service2 via ZeroMQ!
vagrant-service1-1  | ✅ Service1 listening on tcp://*:5555...
vagrant-service1-1  |  Received: Hello World from Service2 via ZeroMQ!
🛑 Para detener los contenedores, presiona Ctrl+C y luego ejecuta: docker-compose down
┌<▸> ~/C/test-zeromq-docker 
└➤                                                                                                                                                                                             
```
---

## Próximos Pasos

* Integrar el intercambio de **mensajes Protobuf compilados**.
* Añadir tests automáticos para verificar la comunicación Protobuf.
* Mejorar la orquestación Docker con `docker-compose` y scripts de inicialización.

---

## Notas

* Este laboratorio está diseñado para un **entorno de prueba controlado**.
* La comunicación actual es **texto plano** mediante ZeroMQ.
* No está optimizado para producción ni incluye cifrado.

