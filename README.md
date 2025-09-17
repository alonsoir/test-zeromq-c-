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

Desde la raíz del proyecto, usar `Makefile` para compilar localmente:

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

