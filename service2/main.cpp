#include <zmq.hpp>
#include <iostream>
#include <string>

int main() {
    zmq::context_t context{1};
    zmq::socket_t socket{context, zmq::socket_type::push};
    socket.connect("tcp://service1:5555");  // En lugar de 127.0.0.1
    std::string message = "Hello World from Service2 via ZeroMQ!";
    zmq::message_t zmq_msg(message.data(), message.size());
    socket.send(zmq_msg, zmq::send_flags::none);

    std::cout << " Sent: " << message << '\n';

    return 0;
}
