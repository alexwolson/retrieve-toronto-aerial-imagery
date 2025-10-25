#!/usr/bin/env python3
"""
Unit tests for fetch_imagery.py
"""

import unittest
from pathlib import Path
import sys

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from fetch_imagery import (
    latlon_to_tile,
    get_tile_bounds,
    compute_tile_indices,
    filter_tiles_by_osm,
    OSMFilter,
    DEFAULT_BBOX
)


class TestTileCalculations(unittest.TestCase):
    """Test tile coordinate calculations."""
    
    def test_latlon_to_tile_zoom_0(self):
        """Test conversion at zoom level 0."""
        x, y = latlon_to_tile(0, 0, 0)
        self.assertEqual(x, 0)
        self.assertEqual(y, 0)
    
    def test_latlon_to_tile_toronto(self):
        """Test conversion for Toronto coordinates."""
        # Toronto City Hall: 43.6532° N, 79.3832° W
        x, y = latlon_to_tile(43.6532, -79.3832, 10)
        # At zoom 10, should get reasonable tile indices
        self.assertGreater(x, 0)
        self.assertGreater(y, 0)
        self.assertLess(x, 1024)  # 2^10
        self.assertLess(y, 1024)
    
    def test_get_tile_bounds(self):
        """Test tile bounds calculation."""
        min_x, min_y, max_x, max_y = get_tile_bounds(0, 0, 0)
        
        # At zoom 0, tile 0,0 should cover entire Web Mercator extent
        web_mercator_extent = 20037508.342789244
        self.assertAlmostEqual(min_x, -web_mercator_extent, places=2)
        self.assertAlmostEqual(max_x, web_mercator_extent, places=2)
        self.assertAlmostEqual(min_y, -web_mercator_extent, places=2)
        self.assertAlmostEqual(max_y, web_mercator_extent, places=2)
    
    def test_compute_tile_indices_small_area(self):
        """Test tile index computation for small area."""
        # Small area around Toronto City Hall
        bbox = [-79.39, 43.65, -79.38, 43.66]
        zoom = 10
        
        tiles = compute_tile_indices(bbox, zoom)
        
        # Should get at least 1 tile
        self.assertGreater(len(tiles), 0)
        
        # All tiles should be tuples of two integers
        for tile in tiles:
            self.assertIsInstance(tile, tuple)
            self.assertEqual(len(tile), 2)
            self.assertIsInstance(tile[0], int)
            self.assertIsInstance(tile[1], int)
    
    def test_compute_tile_indices_default_bbox(self):
        """Test tile index computation for default Toronto bbox."""
        zoom = 10
        tiles = compute_tile_indices(DEFAULT_BBOX, zoom)
        
        # Default Toronto area should produce multiple tiles at zoom 10
        self.assertGreater(len(tiles), 1)
        
        # Tiles should be unique
        self.assertEqual(len(tiles), len(set(tiles)))
    
    def test_tile_indices_increase_with_zoom(self):
        """Test that higher zoom levels produce more tiles."""
        # Use larger area to ensure multiple tiles at each zoom
        bbox = [-79.5, 43.6, -79.3, 43.7]
        
        tiles_zoom_8 = compute_tile_indices(bbox, 8)
        tiles_zoom_10 = compute_tile_indices(bbox, 10)
        tiles_zoom_12 = compute_tile_indices(bbox, 12)
        
        # Higher zoom should produce more or equal tiles
        self.assertLessEqual(len(tiles_zoom_8), len(tiles_zoom_10))
        self.assertLessEqual(len(tiles_zoom_10), len(tiles_zoom_12))


class TestWMTSClient(unittest.TestCase):
    """Test WMTS client functionality."""
    
    def test_ortho_layer_pattern_matching(self):
        """Test that ortho layer pattern matching works."""
        import re
        ortho_pattern = re.compile(r'(cot_)?ortho[_-]?(\d{4})', re.IGNORECASE)
        
        # Test various layer name formats
        test_cases = [
            ('cot_ortho_2022_color_8cm', True, 2022),
            ('ortho_2021_color', True, 2021),
            ('Ortho-2020', True, 2020),
            ('ORTHO2019', True, 2019),
            ('basemap_2022', False, None),
            ('topo_map', False, None),
        ]
        
        for layer_name, should_match, expected_year in test_cases:
            match = ortho_pattern.search(layer_name)
            if should_match:
                self.assertIsNotNone(match, f"Pattern should match {layer_name}")
                self.assertEqual(int(match.group(2)), expected_year)
            else:
                self.assertIsNone(match, f"Pattern should not match {layer_name}")


class TestTileDownloader(unittest.TestCase):
    """Test tile downloader functionality."""
    
    def test_cache_path_generation(self):
        """Test that cache paths are generated consistently."""
        from fetch_imagery import TileDownloader
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            downloader = TileDownloader(Path(tmpdir))
            
            url1 = "https://example.com/tile/1/2/3.png"
            url2 = "https://example.com/tile/1/2/4.png"
            
            path1 = downloader._get_cache_path(url1)
            path2 = downloader._get_cache_path(url2)
            
            # Same URL should give same path
            self.assertEqual(path1, downloader._get_cache_path(url1))
            
            # Different URLs should give different paths
            self.assertNotEqual(path1, path2)
            
            # Paths should be in cache directory
            self.assertTrue(str(path1).startswith(tmpdir))
            self.assertTrue(str(path2).startswith(tmpdir))
    
    def test_cache_format_selection(self):
        """Test that different cache formats are supported."""
        from fetch_imagery import TileDownloader
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test WebP format (default)
            downloader_webp = TileDownloader(Path(tmpdir), cache_format='webp')
            self.assertEqual(downloader_webp.cache_format, 'webp')
            path_webp = downloader_webp._get_cache_path("https://example.com/tile.png")
            self.assertTrue(str(path_webp).endswith('.webp'))
            
            # Test PNG format
            downloader_png = TileDownloader(Path(tmpdir), cache_format='png')
            self.assertEqual(downloader_png.cache_format, 'png')
            path_png = downloader_png._get_cache_path("https://example.com/tile.png")
            self.assertTrue(str(path_png).endswith('.png'))
            
            # Test JPEG format
            downloader_jpg = TileDownloader(Path(tmpdir), cache_format='jpeg')
            self.assertEqual(downloader_jpg.cache_format, 'jpg')  # Should normalize to 'jpg'
            path_jpg = downloader_jpg._get_cache_path("https://example.com/tile.png")
            self.assertTrue(str(path_jpg).endswith('.jpg'))
    
    def test_invalid_cache_format_defaults_to_webp(self):
        """Test that invalid cache format defaults to webp."""
        from fetch_imagery import TileDownloader
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            downloader = TileDownloader(Path(tmpdir), cache_format='invalid')
            self.assertEqual(downloader.cache_format, 'webp')
    
    def test_find_existing_cache(self):
        """Test finding cache files in different formats."""
        from fetch_imagery import TileDownloader
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            downloader = TileDownloader(Path(tmpdir), cache_format='webp')
            
            url = "https://example.com/tile/1/2/3.png"
            url_hash = __import__('hashlib').md5(url.encode()).hexdigest()
            
            # Create a PNG cache file
            png_cache = Path(tmpdir) / f"{url_hash}.png"
            png_cache.touch()
            
            # Should find the PNG file even though format is webp
            found = downloader._find_existing_cache(url)
            self.assertIsNotNone(found)
            self.assertEqual(found, png_cache)
            
            # Test with no cache
            url2 = "https://example.com/tile/4/5/6.png"
            found2 = downloader._find_existing_cache(url2)
            self.assertIsNone(found2)


class TestDryRunFeature(unittest.TestCase):
    """Test the dry-run estimation feature."""
    
    def test_format_bytes(self):
        """Test byte formatting."""
        from fetch_imagery import format_bytes
        
        self.assertEqual(format_bytes(100), "100.00 B")
        self.assertEqual(format_bytes(1024), "1.00 KB")
        self.assertEqual(format_bytes(1024 * 1024), "1.00 MB")
        self.assertEqual(format_bytes(1024 * 1024 * 1024), "1.00 GB")
    
    def test_format_time(self):
        """Test time formatting."""
        from fetch_imagery import format_time
        
        self.assertEqual(format_time(30), "30 seconds")
        self.assertEqual(format_time(90), "1.5 minutes")
        self.assertEqual(format_time(3600), "1.0 hours")
        self.assertEqual(format_time(7200), "2.0 hours")


class TestOSMFiltering(unittest.TestCase):
    """Test OSM filtering functionality."""
    
    def test_osm_filter_initialization(self):
        """Test OSM filter can be initialized."""
        bbox = [-79.5, 43.6, -79.3, 43.7]
        osm_filter = OSMFilter(bbox)
        self.assertEqual(osm_filter.bbox, bbox)
        self.assertEqual(len(osm_filter.road_geometries), 0)
    
    def test_line_intersects_bbox_point_inside(self):
        """Test line intersection when point is inside bbox."""
        osm_filter = OSMFilter([0, 0, 1, 1])
        
        # Line with point inside bbox
        coords = [(0.5, 0.5), (1.5, 1.5)]
        result = osm_filter._line_intersects_bbox(coords, 0, 0, 1, 1)
        self.assertTrue(result)
    
    def test_line_intersects_bbox_point_outside(self):
        """Test line intersection when all points are far outside bbox."""
        osm_filter = OSMFilter([0, 0, 1, 1])
        
        # Line completely outside bbox (beyond margin)
        coords = [(5, 5), (6, 6)]
        result = osm_filter._line_intersects_bbox(coords, 0, 0, 1, 1)
        self.assertFalse(result)
    
    def test_line_intersects_bbox_nearby(self):
        """Test line intersection when points are nearby (within margin)."""
        osm_filter = OSMFilter([0, 0, 1, 1])
        
        # Line just outside bbox but within margin
        coords = [(1.05, 1.05), (1.08, 1.08)]
        result = osm_filter._line_intersects_bbox(coords, 0, 0, 1, 1)
        self.assertTrue(result)
    
    def test_tile_intersects_roads_with_geometries(self):
        """Test tile intersection with manually added road geometries."""
        bbox = [-79.5, 43.6, -79.3, 43.7]
        osm_filter = OSMFilter(bbox)
        
        # Manually add a road geometry (simulating a road in Toronto)
        # This road runs through the approximate center of our bbox
        osm_filter.road_geometries = [
            [(-79.4, 43.65), (-79.35, 43.65)]  # Horizontal road
        ]
        
        # Get a tile in the middle of the bbox at zoom 10
        col, row = latlon_to_tile(43.65, -79.4, 10)
        
        # This tile should intersect with the road
        result = osm_filter.tile_intersects_roads(col, row, 10)
        self.assertTrue(result)
        
        # Get a tile far from the road
        col_far, row_far = latlon_to_tile(43.8, -79.6, 10)
        result_far = osm_filter.tile_intersects_roads(col_far, row_far, 10)
        self.assertFalse(result_far)
    
    def test_tile_intersects_roads_caching(self):
        """Test that tile intersection results are cached."""
        bbox = [-79.5, 43.6, -79.3, 43.7]
        osm_filter = OSMFilter(bbox)
        osm_filter.road_geometries = [[(-79.4, 43.65), (-79.35, 43.65)]]
        
        col, row = latlon_to_tile(43.65, -79.4, 10)
        
        # First call
        result1 = osm_filter.tile_intersects_roads(col, row, 10)
        
        # Check cache was populated
        self.assertIn((col, row, 10), osm_filter._cache)
        
        # Second call should use cache
        result2 = osm_filter.tile_intersects_roads(col, row, 10)
        
        # Results should be the same
        self.assertEqual(result1, result2)


if __name__ == '__main__':
    unittest.main()
