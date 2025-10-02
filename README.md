# retrieve-toronto-aerial-imagery

A Python CLI tool to fetch City of Toronto orthorectified aerial imagery from the public OGC WMTS (Web Map Tile Service) endpoint and create Cloud-Optimized GeoTIFF mosaics.

## Features

- **Auto-detection**: Automatically detects the newest ortho layer (e.g., `cot_ortho_2022_color_8cm`)
- **High Resolution**: Picks the highest available zoom level by default
- **Smart Tile Management**: Computes tile indices for Toronto extent automatically
- **OSM-Based Filtering**: Reduce dataset size by keeping only tiles with roads/sidewalks from OpenStreetMap (new!)
- **Dry Run Mode**: Estimate download size and time before downloading
- **Rich Progress Display**: Beautiful progress bars and enhanced logging powered by Rich
- **Concurrent Downloads**: Downloads tiles concurrently with configurable worker threads
- **Retry Logic**: Built-in retry mechanism with exponential backoff for failed downloads
- **Resume-Safe Cache**: Caches downloaded tiles to disk, allowing for resumable downloads
- **Cloud-Optimized GeoTIFF**: Outputs COG format for efficient cloud storage and access
- **EPSG:3857 Support**: Works with Web Mercator projection (EPSG:3857)

## Installation

### Requirements

- Python 3.9 or higher
- GDAL libraries installed on your system
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Install GDAL

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install gdal-bin libgdal-dev
```

**macOS (with Homebrew):**
```bash
brew install gdal
```

**Windows:**
- Download and install from [OSGeo4W](https://trac.osgeo.org/osgeo4w/)

### Install uv (Recommended)

**Install uv:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Install Python Dependencies

**Option 1: Using uv (Recommended)**
```bash
# Clone the repository
git clone https://github.com/your-username/retrieve-toronto-aerial-imagery.git
cd retrieve-toronto-aerial-imagery

# Install dependencies and the package
uv sync

# Run the tool directly with uv
uv run fetch-toronto-imagery

# Or run the Python script directly
uv run python fetch_imagery.py
```

**Option 2: Using pip with requirements.txt**
```bash
pip install -r requirements.txt
```

**Option 3: Using pip with setup.py (installs as a command)**
```bash
pip install -e .
# Now you can use: fetch-toronto-imagery instead of: python fetch_imagery.py
```

## Usage

### Basic Usage

Fetch aerial imagery for Toronto with default settings:

**Using uv (Recommended):**
```bash
uv run python fetch_imagery.py
# or
uv run fetch-toronto-imagery
```

**Using pip:**
```bash
python fetch_imagery.py
# or (if installed with pip install -e .)
fetch-toronto-imagery
```

This will:
- Auto-detect the newest ortho layer
- Use the highest available zoom level
- Download tiles for the default Toronto extent
- Save output to `toronto_aerial.tif`

### Advanced Usage

#### Dry Run - Estimate Download Size

Before downloading, you can estimate the total download size and time:

```bash
# Using uv
uv run python fetch_imagery.py --bbox -79.4 43.64 -79.36 43.67 --zoom 19 --dry-run

# Using pip
python fetch_imagery.py --bbox -79.4 43.64 -79.36 43.67 --zoom 19 --dry-run
```

This will:
- Sample a few tiles to estimate average size
- Check which tiles are already cached
- Show estimated download size and time for different connection speeds
- Exit without downloading anything

**Example output:**
```
📊 Download Estimation Summary
================================================================================
Bounding box:          [-79.4, 43.64, -79.36, 43.67]
Zoom level:            19
Total tiles:           256
Already cached:        0 tiles (0.00 B)
Tiles to download:     256

Average tile size:     87.34 KB
                       (based on 10 sampled tiles)

📥 Estimated download: 21.84 MB
💾 Total size:         21.84 MB

⏱️  Time Estimates (approximate):

  Fast (50 Mbps)      : ~4 seconds
  Medium (10 Mbps)    : ~18 seconds
  Slow (2 Mbps)       : ~87 seconds

💡 Note: Actual download time depends on server response time,
   network conditions, and concurrent workers (--max-workers).
```

#### Specify Custom Bounding Box

```bash
# Using uv
uv run python fetch_imagery.py --bbox -79.5 43.6 -79.3 43.7

# Using pip
python fetch_imagery.py --bbox -79.5 43.6 -79.3 43.7
```

Bounding box format: `WEST SOUTH EAST NORTH` in EPSG:4326 (WGS84) coordinates

#### Override Layer Name

```bash
# Using uv
uv run python fetch_imagery.py --layer cot_ortho_2022_color_8cm

# Using pip
python fetch_imagery.py --layer cot_ortho_2022_color_8cm
```

#### Specify Zoom Level

```bash
# Using uv
uv run python fetch_imagery.py --zoom 18

# Using pip
python fetch_imagery.py --zoom 18
```

#### Filter by OpenStreetMap Road/Sidewalk Features (NEW!)

Reduce dataset size by only downloading tiles that contain roads or sidewalks:

```bash
# Using uv
uv run python fetch_imagery.py --filter-osm --bbox -79.5 43.6 -79.3 43.7

# Using pip
python fetch_imagery.py --filter-osm --bbox -79.5 43.6 -79.3 43.7
```

This feature:
- Queries OpenStreetMap (OSM) via the Overpass API for road and sidewalk features in your bounding box
- Filters out tiles that don't intersect with any street-level features
- Can significantly reduce the dataset size (often by 50-80% in suburban/rural areas)
- Useful when you only need imagery of streets and their immediate surroundings

**Note:** The `--filter-osm` option requires internet access to query the Overpass API and may add 1-2 minutes to processing time for large bounding boxes.

#### Adjust Concurrency

```bash
# Using uv
uv run python fetch_imagery.py --max-workers 16

# Using pip
python fetch_imagery.py --max-workers 16
```

#### Custom Output Path

```bash
# Using uv
uv run python fetch_imagery.py --output downtown_toronto.tif

# Using pip
python fetch_imagery.py --output downtown_toronto.tif
```

#### Use Different WMTS URL

```bash
# Using uv
uv run python fetch_imagery.py --wmts-url "https://example.com/wmts/WMTSCapabilities.xml"

# Using pip
python fetch_imagery.py --wmts-url "https://example.com/wmts/WMTSCapabilities.xml"
```

### Complete Example

```bash
# Using uv
uv run python fetch_imagery.py \
  --bbox -79.4 43.64 -79.36 43.67 \
  --zoom 19 \
  --max-workers 12 \
  --cache-dir ./my_cache \
  --output highres_downtown.tif \
  --verbose

# Using pip
python fetch_imagery.py \
  --bbox -79.4 43.64 -79.36 43.67 \
  --zoom 19 \
  --max-workers 12 \
  --cache-dir ./my_cache \
  --output highres_downtown.tif \
  --verbose
```

## Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--wmts-url` | WMTS capabilities URL | Toronto Current Year Orthophotos service |
| `--layer` | Layer name override | Auto-detected newest ortho layer |
| `--bbox WEST SOUTH EAST NORTH` | Bounding box in EPSG:4326 | Toronto extent: -79.639 43.581 -79.116 43.855 |
| `--zoom` | Zoom level | Auto-detected maximum zoom |
| `--max-workers` | Concurrent download threads | 8 |
| `--cache-dir` | Tile cache directory | `.tile_cache` |
| `--output` / `-o` | Output GeoTIFF path | `toronto_aerial.tif` |
| `--tile-matrix-set` | Tile matrix set identifier | `default` |
| `--verbose` / `-v` | Enable verbose logging | False |
| `--dry-run` | Estimate download size without downloading | False |
| `--filter-osm` | Filter tiles to keep only those with roads/sidewalks | False |

## How It Works

1. **Fetch Capabilities**: Retrieves WMTS capabilities XML to discover available layers
2. **Auto-Detection**: Identifies the newest ortho layer based on year patterns in layer names
3. **Tile Calculation**: Computes which tiles are needed to cover the specified bounding box
4. **Concurrent Download**: Downloads tiles in parallel using a thread pool
5. **Caching**: Stores tiles locally to avoid re-downloading (resume-safe)
6. **Mosaicing**: Assembles tiles into a single georeferenced image
7. **COG Creation**: Converts to Cloud-Optimized GeoTIFF format with compression

## Resume-Safe Cache

The tool maintains a local cache of downloaded tiles in the cache directory (default: `.tile_cache`). This means:

- If the download is interrupted, you can re-run the same command and it will resume from where it left off
- Already downloaded tiles are not re-downloaded
- Cache files are named using MD5 hashes of tile URLs
- To force re-download, delete the cache directory or specific cached tiles

## Output Format

The output is a Cloud-Optimized GeoTIFF (COG) with the following characteristics:

- **Projection**: EPSG:3857 (Web Mercator)
- **Compression**: DEFLATE with predictor
- **Format**: COG (optimized for cloud storage and streaming)
- **Tiling**: Internal tiling for efficient access
- **Bands**: 3 (RGB) or 4 (RGBA) depending on source

## Troubleshooting

### Import Error: No module named 'osgeo'

Make sure GDAL is installed on your system and the Python bindings are available:

```bash
pip install gdal==$(gdal-config --version)
```

### Permission Denied Error

Ensure you have write permissions for the output directory and cache directory.

### Network Timeout Errors

Try reducing `--max-workers` to avoid overwhelming the server:

```bash
python fetch_imagery.py --max-workers 4
```

### Large Bounding Box

For large areas, consider:
- Using a lower zoom level
- Splitting the area into smaller chunks
- Increasing timeout values in the code

## Examples

### Downtown Toronto Core

```bash
# Using uv
uv run python fetch_imagery.py \
  --bbox -79.39 43.64 -79.37 43.66 \
  --zoom 20 \
  --output downtown_core.tif

# Using pip
python fetch_imagery.py \
  --bbox -79.39 43.64 -79.37 43.66 \
  --zoom 20 \
  --output downtown_core.tif
```

### Entire City (Low Resolution)

```bash
# Using uv
uv run python fetch_imagery.py \
  --bbox -79.639 43.581 -79.116 43.855 \
  --zoom 14 \
  --output toronto_full_lowres.tif

# Using pip
python fetch_imagery.py \
  --bbox -79.639 43.581 -79.116 43.855 \
  --zoom 14 \
  --output toronto_full_lowres.tif
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Credits

- Data source: City of Toronto Open Data
- Built with: GDAL, Pillow, Requests, PyProj, NumPy