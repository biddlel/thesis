#ifndef AUDIO_FILTER_SPH0645
#define AUDIO_FILTER_SPH0645

#include <Audio.h>


class AudioFilterSPH0645 : public AudioStream {
public:
  AudioFilterSPH0645() 
    : AudioStream(1, inputQueueArray),
      conn1(*this, bq),
      conn2(bq, amp)
  {
    // Section 1
    const double coeff1[5] = {
      1.0, -1, 0, // B0, B1, B2
      -0.9992, 0    // A1, A2
    };
    bq.setCoefficients(0, coeff1);

    // Section 2
    const double coeff2[5] = {
      1.0, -1.988897663539382, 0.98892847900809, // B0, B1, B2
      -1.993853376183491, 0.993862821429572    // A1, A2
    };        
    bq.setCoefficients(1, coeff2);

    // Overall A-weighting gain
    amp.gain(1.4125375f);
  }

  virtual void update(void);

private:
  audio_block_t *inputQueueArray[1];
  AudioFilterBiquad bq;
  AudioAmplifier amp;
  AudioConnection conn1;
  AudioConnection conn2;
};

#endif