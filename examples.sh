#!/bin/bash
# Example usage scripts for fetch_imagery.py

# Example 1: Download imagery for downtown Toronto with default settings
echo "Example 1: Default downtown Toronto area"
python3 fetch_imagery.py

# Example 2: Small area with high resolution
echo "Example 2: High-resolution small area around City Hall"
python3 fetch_imagery.py \
  --bbox -79.39 43.64 -79.37 43.66 \
  --zoom 19 \
  --output downtown_highres.tif \
  --verbose

# Example 3: Larger area with medium resolution
echo "Example 3: Downtown core at medium resolution"
python3 fetch_imagery.py \
  --bbox -79.42 43.63 -79.35 43.68 \
  --zoom 16 \
  --max-workers 12 \
  --output downtown_core.tif

# Example 4: Specific layer override
echo "Example 4: Using specific layer"
python3 fetch_imagery.py \
  --layer cot_ortho_2022_color_8cm \
  --bbox -79.4 43.65 -79.38 43.67 \
  --zoom 18 \
  --output specific_layer.tif

# Example 5: Test run with low resolution (fast)
echo "Example 5: Quick test with low resolution"
python3 fetch_imagery.py \
  --bbox -79.4 43.65 -79.39 43.66 \
  --zoom 12 \
  --max-workers 4 \
  --output test_quick.tif

# Example 6: Full city at low resolution (may take a while)
echo "Example 6: Entire Toronto at low resolution"
python3 fetch_imagery.py \
  --bbox -79.639 43.581 -79.116 43.855 \
  --zoom 13 \
  --max-workers 16 \
  --cache-dir ./toronto_cache \
  --output toronto_full_lowres.tif \
  --verbose
