#pragma once
#include <Arduino.h>
#include <Audio.h>
#include <arm_math.h>
#include "config.h"

class DOAMusic : public AudioStream {
public:
    DOAMusic();
    void begin();
    bool available() const { return _new; }
    float read()           { _new = false; return _azimuthDeg; }
    float confidence() const { return _conf; }
    virtual void update() override;

private:
    void buildCovariance();
    void eigenNoiseSubspace();
    float scanAzimuth();

    // state
    bool   _new = false;
    float  _azimuthDeg = 0;
    float  _conf = 0;
    uint32_t _blkCnt = 0;

    // buffers
    float _hist[NUM_MICS][FFT_SIZE];
    float32_t _R[NUM_MICS*NUM_MICS];
    float32_t _eigVec[NUM_MICS*NUM_MICS];
    float32_t _eigVal[NUM_MICS];

    arm_rfft_fast_instance_f32 _rfft;
    audio_block_t* _inputQ[NUM_MICS];
};
