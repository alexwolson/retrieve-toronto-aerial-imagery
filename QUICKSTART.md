# Quick Start Guide

Get started with the Toronto Aerial Imagery Fetcher in 5 minutes!

## 1. Prerequisites

Ensure you have:
- Python 3.8 or higher
- GDAL system libraries (for rasterio)

### Install GDAL (if not already installed)

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install gdal-bin libgdal-dev
```

**macOS:**
```bash
brew install gdal
```

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Or install as a package:
```bash
pip install -e .
```

## 3. Run Your First Download

### Example 1: Small Test Area (Fast - ~1 minute)
```bash
python3 fetch_imagery.py \
  --bbox -79.4 43.65 -79.39 43.66 \
  --zoom 14 \
  --output test_small.tif
```

### Example 2: Downtown Toronto Core (Medium - ~5-10 minutes)
```bash
python3 fetch_imagery.py \
  --bbox -79.39 43.64 -79.37 43.66 \
  --zoom 17 \
  --max-workers 12 \
  --output downtown.tif \
  --verbose
```

### Example 3: Default Toronto Area (Large - timing varies)
```bash
# This downloads imagery for the entire city at zoom level 14
python3 fetch_imagery.py --zoom 14
```

## 4. Check the Output

Your Cloud-Optimized GeoTIFF will be saved to the output path (default: `toronto_aerial.tif`).

You can view it with:
- QGIS (free GIS software)
- ArcGIS
- Any GeoTIFF-compatible viewer

## 5. Understanding the Cache

Downloaded tiles are cached in `.tile_cache/` by default. This means:
- ✅ Resume downloads if interrupted
- ✅ Avoid re-downloading when adjusting parameters
- ✅ Faster subsequent runs

To clear the cache:
```bash
rm -rf .tile_cache/
```

## 6. Run the Demo

See the tool in action without network access:
```bash
python3 demo_workflow.py
```

## 7. Run Tests

Verify everything works:
```bash
python3 -m unittest test_fetch_imagery -v
```

## Common Issues

### "No module named 'rasterio'"
Install dependencies: `pip install -r requirements.txt`

### "Failed to fetch WMTS capabilities"
Check your internet connection and ensure the WMTS URL is accessible.

### "Out of memory"
Try:
- Smaller bounding box
- Lower zoom level
- Reduce `--max-workers`

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Check [examples.sh](examples.sh) for more usage examples
- Customize parameters for your specific use case

## Getting Help

If you encounter issues:
1. Run with `--verbose` flag for detailed logs
2. Check the issue tracker on GitHub
3. Verify GDAL and dependencies are properly installed
