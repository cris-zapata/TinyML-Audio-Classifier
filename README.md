# FPGA Audio Classification Neural Network Accelerator

This is a project I've been working on to learn more about FPGA design and hardware acceleration for neural networks. The main idea is to take a small audio classification model that I trained in Python and implement the actual inference hardware myself in SystemVerilog.

Right now the model classifies four different sounds:

- clap
- whistle
- speech
- finger snap

The network is pretty small, with a 16 → 8 → 4 architecture. The 16 inputs are spectral features extracted from the audio in Python, followed by 8 hidden neurons and 4 output neurons for the four classes.

## How it works

For the audio side, I split the recordings into 0.5 second windows and use an FFT to extract 16 frequency-band features. Those features are then quantized into integer values that can be used by the FPGA.

The basic flow is:

```text
audio → FFT/features → 16 INT8 values → neural network → predicted sound
```

For the hardware side, I built the neural network using parallel multiply-accumulate datapaths. The first layer processes the 16 input features across 8 neurons, then applies the bias, ReLU, and requantization. The second layer uses 4 neurons to generate the final signed logits, and an argmax selects the predicted class.

I used integer/fixed-point arithmetic instead of floating point since the goal is to eventually run the entire inference engine directly on an FPGA.

## Results so far

The final quantized Python model reached **92.31% accuracy** on a held-out test set of 78 samples.

I also wrote a SystemVerilog testbench to compare the hardware calculations against the Python model. So far I've tested one held-out example from each class, and all four matched exactly at the hidden layer, output logits, and final predicted class.

```text
Clap     → PASS
Whistle  → PASS
Speech   → PASS
Snap     → PASS

Tests passed: 4/4
```

I ran these back-to-back without resetting the accelerator between each inference, which also helped verify that the control logic can restart correctly for multiple inputs.

## Files

The main RTL files are in `rtl/`:

```text
nn_classifier.sv
nn_layer1.sv
nn_layer2.sv
layer1_params.sv
layer2_params.sv
```

The testbench is in `sim/`, and the Python scripts I used for feature extraction, loading data, and training are in `python/`.

## What's next

The next big step is getting the design onto my Basys 3. I still need to synthesize and implement everything in Vivado and see what the actual LUT, register, DSP, and timing results look like.

After that I want to add UART so I can do something like:

```text
audio on my computer
        ↓
Python feature extraction
        ↓
16 INT8 features
        ↓
UART
        ↓
Basys 3
        ↓
FPGA neural network
        ↓
classification
```

Eventually I'd like to make the accelerator less fixed and experiment with programmable weights, reusable MAC hardware, and possibly a small systolic array. I also think it would be interesting to take a future version through an RTL-to-GDSII flow and learn more about the ASIC side of accelerator design.

For now, the RTL accelerator and software model are working in simulation, and FPGA deployment is the next step.
