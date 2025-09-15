# SLM (Spatial Light Modulator) Algorithm Implementation

This directory contains the Python implementation of the SLM algorithm described in `algorithm.txt`.

## Files

- `algorithm.txt` - Original algorithm description
- `slm_algorithm.py` - Main implementation of the full algorithm
- `test_slm.py` - Test implementation with smaller matrix for verification
- `CERN-01_event1_nbit16.csv` - Input data file (16-bit binary string)
- `CERN-02_sec1_nbit16.csv` - Input data file (multiple 16-bit binary strings)
- `requirements.txt` - Python dependencies

## Algorithm Description

The algorithm creates a 1024×1280 pixel matrix divided into macropixels:

- **Macropixel structure**: 16 columns × 8 rows of macropixels
- **Each macropixel**: 80×128 pixels (1280÷16 = 80, 1024÷8 = 128)
- **Total macropixels**: 16 × 8 = 128

### Process:

1. **Initialize**: Create 1024×1280 matrix, all values set to 0
2. **Read CERN-01 data**: 16-bit binary string from `CERN-01_event1_nbit16.csv`
   - Apply to ALL macropixel rows
   - Each bit corresponds to a macropixel column
   - '1' → add π, '0' → add 0
3. **Read CERN-02 data**: First 8 values from `CERN-02_sec1_nbit16.csv`
   - Apply to specific macropixel rows (one value per row)
   - Each bit corresponds to a macropixel column
   - '1' → add π, '0' → add 0
4. **Modulo arithmetic**: Ensure 2π = 0
5. **Generate image**: 
   - White pixels = phase value 0
   - Black pixels = phase value π

## Usage

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run Full Algorithm
```bash
python3 slm_algorithm.py
```

### Run Test Version
```bash
python3 test_slm.py
```

## Output

- `slm_result.png` - Final 1024×1280 image (full algorithm)
- `test_slm_result.png` - Test image with smaller matrix
- Console output with statistics and progress information

## Data Format

- **CERN-01**: Single 16-bit binary string (e.g., "1000010001101110")
- **CERN-02**: Multiple 16-bit binary strings, one per line

## Example Results

The algorithm processes binary data to create phase patterns for spatial light modulation, commonly used in optical applications like holography and beam shaping.

### Statistics from Full Run:
- Total pixels: 1,310,720
- White pixels (value=0): 706,560
- Black pixels (value=π): 604,160
- Unique phase values: 2 (0 and π)
- Phase value range: 0.000 to 3.142 radians
