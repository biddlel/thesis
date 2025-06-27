#include <cstdlib>
#include <iostream>
#include <filesystem>
#include <chrono>
#include <thread>
#include <iomanip>
#include <sstream>

int main() {
    const std::string outputDir = "./images";
    std::filesystem::create_directory(outputDir);

    for (int i = 0; i < 5; ++i) {
        auto now = std::chrono::system_clock::now();
        auto now_time_t = std::chrono::system_clock::to_time_t(now);
        auto now_us = std::chrono::duration_cast<std::chrono::microseconds>(
                          now.time_since_epoch()) % 1000000;

        std::stringstream filename;
        filename << outputDir << "/image_"
                 << std::put_time(std::localtime(&now_time_t), "%Y%m%d_%H%M%S")
                 << "_" << std::setw(6) << std::setfill('0') << now_us.count()
                 << ".jpg";

        std::string command = "libcamera-still -n --autofocus-on-capture -o " + filename.str();
        std::cout << "Running: " << command << std::endl;
        int result = std::system(command.c_str());

        if (result != 0) {
            std::cerr << "libcamera-still failed with code " << result << std::endl;
        }

        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    return 0;
}
