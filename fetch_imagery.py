#!/usr/bin/env python3
"""
CLI tool to fetch City of Toronto orthorectified aerial imagery from OGC WMTS.
Auto-detects the newest ortho layer, downloads tiles concurrently, and creates
a Cloud-Optimized GeoTIFF mosaic.
"""

import argparse
import concurrent.futures
import hashlib
import logging
import math
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from xml.etree import ElementTree as ET

import numpy as np
import rasterio
import requests
from PIL import Image
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from rasterio.warp import Resampling
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

# Configure Rich logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    datefmt='[%X]',
    handlers=[RichHandler(rich_tracebacks=True, show_path=False)]
)
logger = logging.getLogger(__name__)

# WMTS namespace for XML parsing
WMTS_NS = {
    'wmts': 'http://www.opengis.net/wmts/1.0',
    'ows': 'http://www.opengis.net/ows/1.1'
}

# Default WMTS URL for City of Toronto
DEFAULT_WMTS_URL = "https://gis.toronto.ca/arcgis/rest/services/basemap/cot_ortho/MapServer/WMTS/1.0.0/WMTSCapabilities.xml"

# Default Toronto extent in EPSG:4326 (WGS84)
DEFAULT_BBOX = [-79.639, 43.581, -79.116, 43.855]  # [west, south, east, north]


class TileDownloader:
    """Handles concurrent tile downloading with retries and caching."""
    
    def __init__(self, cache_dir: Path, max_workers: int = 8, max_retries: int = 3):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Toronto-Aerial-Imagery-Fetcher/1.0'
        })
    
    def _get_cache_path(self, url: str) -> Path:
        """Generate cache file path from URL."""
        url_hash = hashlib.md5(url.encode()).hexdigest()
        return self.cache_dir / f"{url_hash}.png"
    
    def download_tile(self, url: str) -> Optional[Image.Image]:
        """Download a single tile with caching and retries."""
        cache_path = self._get_cache_path(url)
        
        # Check cache first
        if cache_path.exists():
            try:
                return Image.open(cache_path)
            except Exception as e:
                logger.warning(f"Cache corrupted for {url}: {e}")
                cache_path.unlink()
        
        # Download with retries
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                
                # Save to cache
                img = Image.open(response.raw if hasattr(response, 'raw') else 
                                 __import__('io').BytesIO(response.content))
                img.save(cache_path)
                return img
            
            except Exception as e:
                if attempt == self.max_retries - 1:
                    logger.error(f"Failed to download {url} after {self.max_retries} attempts: {e}")
                    return None
                else:
                    logger.warning(f"Attempt {attempt + 1} failed for {url}: {e}")
                    time.sleep(2 ** attempt)  # Exponential backoff
        
        return None
    
    def download_tiles_batch(self, tile_urls: List[Tuple[int, int, str]]) -> Dict[Tuple[int, int], Image.Image]:
        """Download multiple tiles concurrently."""
        results = {}
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed}/{task.total})"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task("Downloading tiles", total=len(tile_urls))
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_tile = {
                    executor.submit(self.download_tile, url): (col, row)
                    for col, row, url in tile_urls
                }
                
                for future in concurrent.futures.as_completed(future_to_tile):
                    col, row = future_to_tile[future]
                    try:
                        img = future.result()
                        if img is not None:
                            results[(col, row)] = img
                    except Exception as e:
                        logger.error(f"Error processing tile ({col}, {row}): {e}")
                    
                    progress.update(task, advance=1)
        
        return results
    
    def _get_tile_size_head(self, url: str) -> Optional[int]:
        """Get tile size using HEAD request."""
        try:
            response = self.session.head(url, timeout=10)
            content_length = response.headers.get('Content-Length')
            if content_length:
                return int(content_length)
        except Exception:
            pass
        return None
    
    def estimate_download_size(self, tile_urls: List[Tuple[int, int, str]], 
                              sample_count: int = 10) -> Dict[str, any]:
        """
        Estimate total download size by sampling tiles.
        Returns statistics about the download.
        """
        import random
        
        total_tiles = len(tile_urls)
        
        # Sample some tiles to estimate size (sample from all tiles without checking cache)
        sample_size = min(sample_count, len(tile_urls))
        sample_urls = random.sample(tile_urls, sample_size)
        
        logger.info(f"Sampling {sample_size} tiles to estimate download size...")
        
        sample_sizes = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task("Sampling tiles", total=sample_size)
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, sample_size)) as executor:
                futures = [executor.submit(self._get_tile_size_head, url) 
                          for _, _, url in sample_urls]
                
                for future in concurrent.futures.as_completed(futures):
                    try:
                        size = future.result()
                        if size:
                            sample_sizes.append(size)
                    except Exception as e:
                        logger.debug(f"Error sampling tile: {e}")
                    
                    progress.update(task, advance=1)
        
        # Calculate average and estimate
        if sample_sizes:
            avg_tile_size = sum(sample_sizes) / len(sample_sizes)
        else:
            # Fallback estimate: aerial imagery tiles are typically 50-150KB
            avg_tile_size = 100_000  # 100KB
        
        estimated_download_size = avg_tile_size * total_tiles
        
        return {
            'total_tiles': total_tiles,
            'cached_tiles': 0,
            'tiles_to_download': total_tiles,
            'total_size_bytes': estimated_download_size,
            'download_size_bytes': estimated_download_size,
            'cached_size_bytes': 0,
            'average_tile_size_bytes': avg_tile_size,
            'sample_count': len(sample_sizes),
            'is_estimate': True
        }


class WMTSClient:
    """Client for parsing WMTS capabilities and generating tile URLs."""
    
    def __init__(self, wmts_url: str):
        self.wmts_url = wmts_url
        self.capabilities = None
        self.layers = {}
    
    def fetch_capabilities(self) -> bool:
        """Fetch and parse WMTS capabilities XML."""
        try:
            logger.info(f"Fetching WMTS capabilities from {self.wmts_url}")
            response = requests.get(self.wmts_url, timeout=30)
            response.raise_for_status()
            
            self.capabilities = ET.fromstring(response.content)
            self._parse_layers()
            return True
        
        except Exception as e:
            logger.error(f"Failed to fetch WMTS capabilities: {e}")
            return False
    
    def _parse_layers(self):
        """Parse available layers from capabilities."""
        # Try different XML structures (standard WMTS and ESRI REST)
        layers_found = self.capabilities.findall('.//wmts:Layer', WMTS_NS)
        
        if not layers_found:
            # Try without namespace for ESRI REST services
            layers_found = self.capabilities.findall('.//Layer')
        
        for layer in layers_found:
            identifier = layer.find('.//ows:Identifier', WMTS_NS)
            if identifier is None:
                identifier = layer.find('.//wmts:Identifier', WMTS_NS)
            if identifier is None:
                identifier = layer.find('.//Identifier')
            
            if identifier is not None:
                layer_id = identifier.text
                self.layers[layer_id] = layer
                logger.info(f"Found layer: {layer_id}")
    
    def get_newest_ortho_layer(self) -> Optional[str]:
        """Auto-detect the newest ortho layer based on naming patterns."""
        # First, try to find layers with year patterns
        ortho_pattern = re.compile(r'(cot_)?ortho[_-]?(\d{4})', re.IGNORECASE)
        
        ortho_layers = {}
        for layer_id in self.layers.keys():
            match = ortho_pattern.search(layer_id)
            if match:
                year = int(match.group(2))
                ortho_layers[layer_id] = year
        
        if ortho_layers:
            # Return layer with highest year
            newest_layer = max(ortho_layers.items(), key=lambda x: x[1])[0]
            logger.info(f"Auto-detected newest ortho layer: {newest_layer} (year: {ortho_layers[newest_layer]})")
            return newest_layer
        
        # If no year pattern found, look for generic ortho layers
        for layer_id in self.layers.keys():
            if 'ortho' in layer_id.lower() or 'imagery' in layer_id.lower():
                logger.info(f"Auto-detected ortho layer: {layer_id}")
                return layer_id
        
        logger.warning("No ortho layers found")
        return None
    
    def get_max_zoom_level(self, layer_id: str) -> int:
        """Get the maximum zoom level for a layer."""
        layer = self.layers.get(layer_id)
        if layer is None:
            logger.warning(f"Layer {layer_id} not found")
            return 18  # Default fallback
        
        # Try to find TileMatrixSet
        tile_matrix_links = layer.findall('.//wmts:TileMatrixSetLink', WMTS_NS)
        if not tile_matrix_links:
            tile_matrix_links = layer.findall('.//TileMatrixSetLink')
        
        max_zoom = 18  # Default
        
        for link in tile_matrix_links:
            tile_matrix_set_elem = link.find('.//wmts:TileMatrixSet', WMTS_NS)
            if tile_matrix_set_elem is None:
                tile_matrix_set_elem = link.find('.//TileMatrixSet')
            
            if tile_matrix_set_elem is not None:
                tile_matrix_set_id = tile_matrix_set_elem.text
                
                # Find the specific TileMatrixSet definition
                # Try with namespace first
                tile_matrix_set_def = self.capabilities.find(
                    f'.//wmts:TileMatrixSet[ows:Identifier="{tile_matrix_set_id}"]',
                    WMTS_NS
                )
                if tile_matrix_set_def is None:
                    # Try without namespace
                    tile_matrix_set_def = self.capabilities.find(
                        f'.//TileMatrixSet[ows:Identifier="{tile_matrix_set_id}"]',
                        WMTS_NS
                    )
                if tile_matrix_set_def is None:
                    # Try with simple Identifier
                    tile_matrix_set_def = self.capabilities.find(
                        f'.//TileMatrixSet[Identifier="{tile_matrix_set_id}"]'
                    )
                
                if tile_matrix_set_def is not None:
                    # Count TileMatrix elements within this specific TileMatrixSet
                    matrices = tile_matrix_set_def.findall('.//wmts:TileMatrix', WMTS_NS)
                    if not matrices:
                        matrices = tile_matrix_set_def.findall('.//TileMatrix')
                    
                    if matrices:
                        max_zoom = len(matrices) - 1
                        break
        
        logger.info(f"Max zoom level for {layer_id}: {max_zoom}")
        return max_zoom
    
    def build_tile_url_template(self, layer_id: str, tile_matrix_set: str = 'default') -> Optional[str]:
        """Build tile URL template for the given layer."""
        # For ESRI REST services, construct URL from base
        if 'arcgis' in self.wmts_url.lower():
            base_url = self.wmts_url.rsplit('/WMTS', 1)[0]
            # ESRI WMTS URL pattern
            return f"{base_url}/WMTS/tile/1.0.0/{layer_id}/{{tile_matrix_set}}/{{zoom}}/{{row}}/{{col}}.png"
        
        # For standard WMTS, try to extract from capabilities
        layer = self.layers.get(layer_id)
        if layer:
            resource_url = layer.find('.//wmts:ResourceURL[@resourceType="tile"]', WMTS_NS)
            if resource_url is not None:
                template = resource_url.get('template')
                if template:
                    return template
        
        # Fallback: construct generic WMTS URL
        base_url = self.wmts_url.rsplit('?', 1)[0].replace('WMTSCapabilities.xml', '')
        return f"{base_url}tile/1.0.0/{layer_id}/{{tile_matrix_set}}/{{zoom}}/{{row}}/{{col}}.png"


def latlon_to_tile(lat: float, lon: float, zoom: int) -> Tuple[int, int]:
    """Convert lat/lon to tile coordinates at given zoom level."""
    n = 2 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return x, y


def get_tile_bounds(col: int, row: int, zoom: int) -> Tuple[float, float, float, float]:
    """Get geographic bounds of a tile in EPSG:3857."""
    n = 2 ** zoom
    
    # Web Mercator bounds
    web_mercator_extent = 20037508.342789244
    tile_size = (2 * web_mercator_extent) / n
    
    min_x = -web_mercator_extent + col * tile_size
    max_x = min_x + tile_size
    max_y = web_mercator_extent - row * tile_size
    min_y = max_y - tile_size
    
    return min_x, min_y, max_x, max_y


def compute_tile_indices(bbox: List[float], zoom: int) -> List[Tuple[int, int]]:
    """Compute all tile indices needed to cover the bounding box."""
    west, south, east, north = bbox
    
    # Convert corners to tile coordinates
    x_min, y_max = latlon_to_tile(north, west, zoom)
    x_max, y_min = latlon_to_tile(south, east, zoom)
    
    # Ensure proper ordering
    if x_min > x_max:
        x_min, x_max = x_max, x_min
    if y_min > y_max:
        y_min, y_max = y_max, y_min
    
    # Generate all tile indices in the range
    tiles = []
    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            tiles.append((x, y))
    
    logger.info(f"Computed {len(tiles)} tiles for bbox {bbox} at zoom {zoom}")
    return tiles


def create_geotiff_mosaic(tiles_data: Dict[Tuple[int, int], Image.Image], 
                          zoom: int, 
                          output_path: Path,
                          tile_size: int = 256) -> bool:
    """Create a Cloud-Optimized GeoTIFF from tiles."""
    if not tiles_data:
        logger.error("No tiles to mosaic")
        return False
    
    # Determine mosaic dimensions
    cols = [col for col, _ in tiles_data.keys()]
    rows = [row for _, row in tiles_data.keys()]
    
    min_col, max_col = min(cols), max(cols)
    min_row, max_row = min(rows), max(rows)
    
    width = (max_col - min_col + 1) * tile_size
    height = (max_row - min_row + 1) * tile_size
    
    logger.info(f"Creating mosaic: {width}x{height} pixels, {len(tiles_data)} tiles")
    
    # Create output array (assuming RGB)
    sample_tile = next(iter(tiles_data.values()))
    if sample_tile.mode == 'RGB':
        bands = 3
    elif sample_tile.mode == 'RGBA':
        bands = 4
    else:
        bands = 3
        
    mosaic = np.zeros((bands, height, width), dtype=np.uint8)
    
    # Place tiles in mosaic
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed}/{task.total})"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("Placing tiles in mosaic", total=len(tiles_data))
        
        for (col, row), img in tiles_data.items():
            x_offset = (col - min_col) * tile_size
            y_offset = (row - min_row) * tile_size
            
            # Convert to numpy array
            img_array = np.array(img.convert('RGB' if bands == 3 else 'RGBA'))
            
            # Place in mosaic (rasterio expects bands, height, width)
            for band_idx in range(bands):
                mosaic[band_idx, y_offset:y_offset + tile_size, x_offset:x_offset + tile_size] = img_array[:, :, band_idx]
            
            progress.update(task, advance=1)
    
    # Get geographic bounds
    min_x, min_y, _, _ = get_tile_bounds(min_col, min_row, zoom)
    _, _, max_x, max_y = get_tile_bounds(max_col, max_row, zoom)
    
    # Create transform
    transform = from_bounds(min_x, min_y, max_x, max_y, width, height)
    
    # Create temporary GeoTIFF
    temp_path = output_path.with_suffix('.tmp.tif')
    
    # Write GeoTIFF with rasterio
    try:
        with rasterio.open(
            str(temp_path),
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=bands,
            dtype=rasterio.uint8,
            crs=CRS.from_epsg(3857),
            transform=transform,
            compress='deflate',
            tiled=True,
            predictor=2
        ) as dst:
            dst.write(mosaic)
        
        logger.info("Creating Cloud-Optimized GeoTIFF...")
        
        # Convert to COG using rasterio
        cog_profile = {
            'driver': 'GTiff',
            'compress': 'deflate',
            'predictor': 2,
            'tiled': True,
            'blockxsize': 512,
            'blockysize': 512,
            'COPY_SRC_OVERVIEWS': 'YES',
            'BIGTIFF': 'IF_SAFER'
        }
        
        # Read temp file and write as COG
        with rasterio.open(str(temp_path)) as src:
            profile = src.profile.copy()
            profile.update(cog_profile)
            
            with rasterio.open(str(output_path), 'w', **profile) as dst:
                dst.write(src.read())
                
                # Add overviews for COG
                overview_factors = [2, 4, 8, 16]
                dst.build_overviews(overview_factors, Resampling.average)
                dst.update_tags(ns='rio_overview', resampling='average')
        
        # Remove temporary file
        temp_path.unlink()
        
        logger.info(f"Cloud-Optimized GeoTIFF created: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create GeoTIFF: {e}")
        if temp_path.exists():
            temp_path.unlink()
        return False


def format_bytes(bytes_val: float) -> str:
    """Format bytes into human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} PB"


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time string."""
    if seconds < 60:
        return f"{seconds:.0f} seconds"
    elif seconds < 3600:
        return f"{seconds/60:.1f} minutes"
    else:
        hours = seconds / 3600
        if hours < 24:
            return f"{hours:.1f} hours"
        else:
            return f"{hours/24:.1f} days"


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Fetch City of Toronto aerial imagery from OGC WMTS and create Cloud-Optimized GeoTIFF',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--wmts-url',
        default=DEFAULT_WMTS_URL,
        help='WMTS capabilities URL'
    )
    
    parser.add_argument(
        '--layer',
        help='Layer name override (auto-detects newest ortho layer if not provided)'
    )
    
    parser.add_argument(
        '--bbox',
        nargs=4,
        type=float,
        metavar=('WEST', 'SOUTH', 'EAST', 'NORTH'),
        default=DEFAULT_BBOX,
        help='Bounding box in EPSG:4326 (WGS84) coordinates'
    )
    
    parser.add_argument(
        '--zoom',
        type=int,
        help='Zoom level (auto-detects maximum if not provided)'
    )
    
    parser.add_argument(
        '--max-workers',
        type=int,
        default=8,
        help='Maximum concurrent download threads'
    )
    
    parser.add_argument(
        '--cache-dir',
        type=Path,
        default=Path('.tile_cache'),
        help='Directory for tile cache'
    )
    
    parser.add_argument(
        '--output',
        '-o',
        type=Path,
        default=Path('toronto_aerial.tif'),
        help='Output GeoTIFF file path'
    )
    
    parser.add_argument(
        '--tile-matrix-set',
        default='default',
        help='Tile matrix set identifier'
    )
    
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Estimate download size without actually downloading tiles'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Initialize WMTS client
    wmts = WMTSClient(args.wmts_url)
    
    if not wmts.fetch_capabilities():
        logger.error("Failed to fetch WMTS capabilities")
        return 1
    
    # Determine layer
    layer_id = args.layer
    if not layer_id:
        layer_id = wmts.get_newest_ortho_layer()
        if not layer_id:
            logger.error("Could not auto-detect ortho layer. Please specify --layer")
            return 1
    
    if layer_id not in wmts.layers:
        logger.error(f"Layer '{layer_id}' not found in WMTS capabilities")
        logger.info(f"Available layers: {', '.join(wmts.layers.keys())}")
        return 1
    
    # Determine zoom level
    zoom = args.zoom
    if zoom is None:
        zoom = wmts.get_max_zoom_level(layer_id)
    
    logger.info(f"Using layer: {layer_id}, zoom: {zoom}")
    
    # Compute tile indices
    tiles = compute_tile_indices(args.bbox, zoom)
    
    if not tiles:
        logger.error("No tiles to download")
        return 1
    
    # Build tile URL template
    url_template = wmts.build_tile_url_template(layer_id, args.tile_matrix_set)
    if not url_template:
        logger.error("Could not determine tile URL template")
        return 1
    
    logger.info(f"Tile URL template: {url_template}")
    
    # Generate tile URLs
    tile_urls = []
    for col, row in tiles:
        url = url_template.format(
            tile_matrix_set=args.tile_matrix_set,
            zoom=zoom,
            col=col,
            row=row
        )
        tile_urls.append((col, row, url))
    
    # Dry run mode - estimate download size
    if args.dry_run:
        logger.info("=" * 80)
        logger.info("DRY RUN MODE - Estimating download requirements")
        logger.info("=" * 80)
        
        downloader = TileDownloader(args.cache_dir, args.max_workers)
        stats = downloader.estimate_download_size(tile_urls, sample_count=10)
        
        # Format output
        print()
        print("📊 Download Estimation Summary")
        print("=" * 80)
        print(f"Bounding box:          [{args.bbox[0]}, {args.bbox[1]}, {args.bbox[2]}, {args.bbox[3]}]")
        print(f"Zoom level:            {zoom}")
        print(f"Total tiles:           {stats['total_tiles']:,}")
        print(f"Already cached:        {stats['cached_tiles']:,} tiles ({format_bytes(stats['cached_size_bytes'])})")
        print(f"Tiles to download:     {stats['tiles_to_download']:,}")
        print()
        
        if stats['tiles_to_download'] > 0:
            print(f"Average tile size:     {format_bytes(stats['average_tile_size_bytes'])}")
            if stats['is_estimate'] and stats['sample_count'] > 0:
                print(f"                       (based on {stats['sample_count']} sampled tiles)")
            elif not stats['is_estimate']:
                print(f"                       (based on cached tiles)")
            print()
            print(f"📥 Estimated download: {format_bytes(stats['download_size_bytes'])}")
        
        print(f"💾 Total size:         {format_bytes(stats['total_size_bytes'])}")
        print()
        
        # Time estimates
        if stats['tiles_to_download'] > 0:
            print("⏱️  Time Estimates (approximate):")
            print()
            
            # Different connection speeds
            speeds = {
                'Fast (50 Mbps)': 50 * 1024 * 1024 / 8,     # bytes per second
                'Medium (10 Mbps)': 10 * 1024 * 1024 / 8,
                'Slow (2 Mbps)': 2 * 1024 * 1024 / 8
            }
            
            for speed_name, bytes_per_sec in speeds.items():
                seconds = stats['download_size_bytes'] / bytes_per_sec
                time_str = format_time(seconds)
                print(f"  {speed_name:20s}: ~{time_str}")
            
            print()
            print("💡 Note: Actual download time depends on server response time,")
            print("   network conditions, and concurrent workers (--max-workers).")
        else:
            print("✅ All tiles are already cached! No download needed.")
        
        print()
        print("To proceed with the download, run again without --dry-run")
        print("=" * 80)
        
        return 0
    
    # Download tiles
    downloader = TileDownloader(args.cache_dir, args.max_workers)
    
    logger.info(f"Downloading {len(tile_urls)} tiles...")
    tiles_data = downloader.download_tiles_batch(tile_urls)
    
    if not tiles_data:
        logger.error("No tiles downloaded successfully")
        return 1
    
    logger.info(f"Successfully downloaded {len(tiles_data)} / {len(tile_urls)} tiles")
    
    # Create mosaic
    if create_geotiff_mosaic(tiles_data, zoom, args.output):
        logger.info(f"Success! Output: {args.output}")
        return 0
    else:
        logger.error("Failed to create mosaic")
        return 1


if __name__ == '__main__':
    sys.exit(main())
