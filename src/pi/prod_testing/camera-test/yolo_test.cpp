#include <libcamera/libcamera.h>
#include <libcamera/camera_manager.h>
#include <libcamera/framebuffer_allocator.h>
#include <libcamera/request.h>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#include <iostream>
#include <vector>
#include <memory>
#include <thread>
#include <chrono>
#include <fcntl.h>
#include <sys/mman.h>

using namespace libcamera;

class Camera {
public:
    bool init() {
        manager = std::make_unique<CameraManager>();
        if (manager->start()) return false;

        if (manager->cameras().empty()) return false;

        cam = manager->cameras()[0];
        cam->acquire();

        config = cam->generateConfiguration({ StreamRole::Viewfinder });
        config->at(0).pixelFormat = formats::YUV420;
        config->at(0).size = {640, 640};

        if (cam->configure(config.get()) < 0) return false;

        allocator = std::make_unique<FrameBufferAllocator>(cam);
        for (StreamConfiguration &cfg : *config)
            if (allocator->allocate(cfg.stream()) < 0) return false;

        for (StreamConfiguration &cfg : *config) {
            for (const std::unique_ptr<FrameBuffer> &buffer : allocator->buffers(cfg.stream())) {
                std::unique_ptr<Request> request = cam->createRequest();
                request->addBuffer(cfg.stream(), buffer.get());
                requests.push_back(std::move(request));
            }
        }

        cam->start();
        return true;
    }

    cv::Mat capture() {
        Stream *stream = config->at(0).stream();
        Request *request = requests[0].get();
        cam->queueRequest(request);

        Request *completed = nullptr;
        while (!completed) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            if (request->status() == Request::RequestComplete)
                completed = request;
        }

        FrameBuffer *buffer = completed->buffers().at(stream);
        const FrameBuffer::Plane &plane = buffer->planes()[0];
        void *mem = mmap(nullptr, plane.length, PROT_READ, MAP_SHARED, plane.fd.get(), 0);
        if (mem == MAP_FAILED) throw std::runtime_error("Failed to mmap");

        int width = config->at(0).size.width;
        int height = config->at(0).size.height;
        cv::Mat yuv(height + height / 2, width, CV_8UC1, mem);
        cv::Mat bgr;
        cv::cvtColor(yuv, bgr, cv::COLOR_YUV2BGR_I420);

        munmap(mem, plane.length);
        return bgr;
    }

    ~Camera() {
        if (cam) {
            cam->stop();
            cam->release();
        }
        manager->stop();
    }

private:
    std::unique_ptr<CameraManager> manager;
    std::shared_ptr<Camera> cam;
    std::unique_ptr<CameraConfiguration> config;
    std::unique_ptr<FrameBufferAllocator> allocator;
    std::vector<std::unique_ptr<Request>> requests;
};

std::vector<std::string> loadClassNames(const std::string &filename) {
    std::vector<std::string> classNames;
    std::ifstream file(filename);
    std::string line;
    while (std::getline(file, line)) {
        classNames.push_back(line);
    }
    return classNames;
}

void runYolo(cv::Mat &frame, const std::string &modelPath, const std::vector<std::string> &classNames) {
    cv::dnn::Net net = cv::dnn::readNetFromONNX(modelPath);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    cv::Mat blob = cv::dnn::blobFromImage(frame, 1 / 255.0, {640, 640}, {}, true, false);
    net.setInput(blob);

    std::vector<cv::Mat> outputs;
    net.forward(outputs);
    cv::Mat output = outputs[0];

    float confThreshold = 0.4, nmsThreshold = 0.5;
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for (int i = 0; i < output.rows; ++i) {
        float *data = (float *)output.ptr(i);
        float conf = data[4];
        if (conf < confThreshold) continue;

        int classId = std::max_element(data + 5, data + output.cols) - (data + 5);
        float classScore = data[5 + classId];
        if (classScore < confThreshold) continue;

        int cx = static_cast<int>(data[0] * frame.cols);
        int cy = static_cast<int>(data[1] * frame.rows);
        int w = static_cast<int>(data[2] * frame.cols);
        int h = static_cast<int>(data[3] * frame.rows);
        int x = cx - w / 2;
        int y = cy - h / 2;

        boxes.emplace_back(x, y, w, h);
        confidences.push_back(classScore);
        classIds.push_back(classId);
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);

    for (int idx : indices) {
        cv::Rect box = boxes[idx];
        cv::rectangle(frame, box, cv::Scalar(0, 255, 0), 2);
        std::string label = classNames.empty() ? std::to_string(classIds[idx])
                                               : classNames[classIds[idx]];
        cv::putText(frame, label, box.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(255, 0, 0), 2);
    }

    // Save result
    std::string outputFilename = "output_with_detections.jpg";
    cv::imwrite(outputFilename, frame);
    std::cout << "✅ Saved annotated image as: " << outputFilename << std::endl;
    cv::imshow("YOLOv10 Results", frame);
    cv::waitKey(0);
}

int main() {
    try {
        Camera cam;
        if (!cam.init()) throw std::runtime_error("Failed to initialize libcamera");

        std::vector<std::string> classNames = loadClassNames("coco.names");
        cv::Mat frame = cam.capture();
        runYolo(frame, "yolov10n.onnx", classNames);
    } catch (const std::exception &e) {
        std::cerr << "❌ Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}