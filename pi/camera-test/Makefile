# Makefile for Raspberry Pi Camera Capture project

# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++17 -O2 -Wall

# Include and library paths
INCLUDES = -I/usr/include/libcamera
LIBS = -lcamera -lcamera-base -lpthread `pkg-config --cflags --libs opencv4`

TARGETS := camera_capture detect

SRC_camera_capture := camera_capture.cpp
SRC_detect := yolo_test.cpp

all: $(TARGETS)

camera_capture: $(SRC_camera_capture)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $^ -lcamera -lpthread $(LIBS)

detect: $(SRC_detect)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $^ -lcamera -lpthread $(LIBS)

clean:
	rm -f $(TARGETS)