#ifndef AUDIO_FILTER_A_WEIGHT
#define AUDIO_FILTER_A_WEIGHT

#include <Audio.h>

class AudioFilterAWeight : public AudioStream {
public:
  AudioFilterAWeight() 
      : AudioStream(1, inputQueueArray),
      conn1(*this, bq),
      conn2(bq, amp)
  {
    // Section 1
    const double coeff1[5] = {
      1.0,-2.00026996133106, 1.00027056142719,    // B0, B1, B2
      1.060868438509278, 0.163987445885926        // A1, A2
    };
    bq.setCoefficients(0, coeff1);

    // Section 2
    const double coeff2[5] = {
      1.0, 4.35912384203144, 3.09120265783884,    // B0, B1, B2
      -1.208419926363593, 0.273166998428332       // A1, A2
    };
    bq.setCoefficients(1, coeff2);

    // Section 3
    const double coeff3[5] = {
      1.0, -0.70930303489759, -0.29071868393580,    // B0, B1, B2
      -1.982242159753048, 0.982298594928989      // A1, A2
    };
    bq.setCoefficients(2, coeff3);

    // Overall A-weighting gain
    amp.gain(0.169994948147430f);
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