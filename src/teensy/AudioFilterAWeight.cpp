#include <Arduino.h>
#include "AudioFilterAWeight.h"

#if defined(__ARM_ARCH_7EM__)

void AudioFilterAWeight::update(void)
{
  audio_block_t *block;
  block = receiveWritable();
  if (!block) return;
  transmit(block);
	release(block);
}

#endif