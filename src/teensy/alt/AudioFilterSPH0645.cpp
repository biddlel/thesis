#include <Arduino.h>
#include "AudioFilterSPH0645.h"

#if defined(__ARM_ARCH_7EM__)

void AudioFilterSPH0645::update(void)
{
  audio_block_t *block;
  block = receiveWritable();
  if (!block) return;
  transmit(block);
	release(block);
}

#endif