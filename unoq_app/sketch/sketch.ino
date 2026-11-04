// AirPen Matrix, MCU side of the UNO Q (STM32U585).
// Receives 104-byte grayscale frames (8 rows x 13 columns, row-major, 0..255)
// from the Linux side over the Router Bridge and shows them on the LED matrix.
#include <Arduino_RouterBridge.h>
#include <Arduino_LED_Matrix.h>
#include <vector>

Arduino_LED_Matrix matrix;

static const size_t FRAME_BYTES = 8 * 13;

void draw(std::vector<uint8_t> frame) {
  if (frame.size() != FRAME_BYTES) {
    return;  // ignore malformed frames rather than reading past the buffer
  }
  matrix.draw(frame.data());
}

void setup() {
  matrix.begin();
  matrix.setGrayscaleBits(8);  // accept 0..255 brightness per LED
  Bridge.begin();
  Bridge.provide("draw", draw);
}

void loop() {
  delay(5);
}
