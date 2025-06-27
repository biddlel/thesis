#include <libcamera/libcamera.h>
#include <libcamera/camera_manager.h>
#include <libcamera/framebuffer_allocator.h>
#include <libcamera/request.h>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

using namespace libcamera;

class Camera {
public:
    bool init();
    cv:Mat capture();
    ~Camera();
private:
    std::unique_ptr<CameraManager> manager;
    std::shared_ptr<Camera> cam;
    std::unique_ptr<CameraConfiguration> config;
    std::unique_ptr<FrameBufferAllocator> allocator;
    std::vector<std::unique_ptr<Request>> requests; 
};