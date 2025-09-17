#include <zmq.hpp>
#include <iostream>
#include <string>

int main() {
    zmq::context_t context{1};
    zmq::socket_t socket{context, zmq::socket_type::pull};
    socket.bind("tcp://*:5555");

    std::cout << "✅ Service1 listening on tcp://*:5555...\n";

    while (true) {
        zmq::message_t message;
        socket.recv(message, zmq::recv_flags::none);
        std::string msg_str(static_cast<char*>(message.data()), message.size());
        std::cout << " Received: " << msg_str << '\n';
        break; // Para POC de Hello World, recibimos uno y salimos
    }

    return 0;
}
