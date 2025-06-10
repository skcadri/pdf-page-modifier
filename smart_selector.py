"""
Smart Selection Module for PDF Page Modifier

This module implements ML-based pattern matching for automatic page selection.
It provides content type detection, example-based selection, and quick presets.
"""

import numpy as np
import cv2
import fitz  # PyMuPDF
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum
import threading
import logging
from PIL import Image
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from scipy import ndimage
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContentType(Enum):
    """Enumeration of different page content types"""
    TEXT_HEAVY = "text_heavy"
    IMAGE_HEAVY = "image_heavy"
    SINGLE_IMAGE = "single_image"  # For pages with just one main image
    MIXED = "mixed"
    BLANK = "blank"
    TABLE = "table"
    TITLE = "title"
    CHART = "chart"

@dataclass
class TextBlock:
    """Represents a text block on a page"""
    x: float
    y: float
    width: float
    height: float
    text: str
    font_size: float

@dataclass
class ImageRegion:
    """Represents an image region on a page"""
    x: float
    y: float
    width: float
    height: float
    area: float

@dataclass
class TextDistribution:
    """Advanced text distribution analysis"""
    text_density_map: np.ndarray  # 2D density grid
    text_clusters: List[Dict]  # Clustered text regions
    vertical_distribution: np.ndarray  # Text density by vertical position
    horizontal_distribution: np.ndarray  # Text density by horizontal position
    text_flow_pattern: str  # "single_column", "multi_column", "scattered", "structured"
    margin_analysis: Dict[str, float]  # top, bottom, left, right margins
    line_spacing_stats: Dict[str, float]  # mean, std, median line spacing
    text_block_sizes: List[float]  # Sizes of major text blocks
    text_regularity_score: float  # How regular/structured the text layout is

@dataclass
class AdvancedLayoutFeatures:
    """Advanced layout analysis for robust page matching"""
    text_blocks: List[TextBlock]
    image_regions: List[ImageRegion]
    white_space_ratio: float
    text_line_count: int
    font_sizes: List[int]
    page_width: float
    page_height: float
    
    # Advanced features
    text_distribution: TextDistribution
    font_variation_score: float  # How much font sizes vary
    text_alignment_pattern: str  # "left", "center", "right", "justified", "mixed"
    content_hierarchy_score: float  # How hierarchical the content structure is
    visual_balance_score: float  # How visually balanced the page is
    reading_complexity_score: float  # How complex the reading pattern is

@dataclass
class PageFeatures:
    """Complete feature set for a page"""
    text_embedding: Optional[np.ndarray]
    text_density: float
    layout_features: AdvancedLayoutFeatures
    content_type: ContentType
    has_tables: bool
    has_images: bool
    is_blank: bool
    text_content: str
    
    # Advanced similarity features
    structure_fingerprint: np.ndarray  # Unique structural signature
    visual_complexity_score: float
    information_density_score: float

@dataclass
class SelectionResult:
    """Result of smart selection operation"""
    selected_pages: List[int]
    confidence_scores: List[float]
    reasoning: str
    features_found: List[str]
    total_pages_analyzed: int
    processing_time: float

class AdvancedFeatureExtractor:
    """Advanced feature extraction for robust page analysis"""
    
    def __init__(self):
        pass  # This class only provides utility methods
        
    def _convert_image_to_numpy(self, page_image) -> Optional[np.ndarray]:
        """Convert PIL Image or other formats to numpy array"""
        try:
            if page_image is None:
                return None
                
            # Handle PIL Image objects
            if hasattr(page_image, 'convert'):
                # Convert PIL Image to numpy array
                if page_image.mode != 'RGB':
                    page_image = page_image.convert('RGB')
                return np.array(page_image)
                
            # Handle numpy arrays
            elif isinstance(page_image, np.ndarray):
                return page_image
                
            # Handle other image objects
            else:
                # Try to convert to PIL first
                if hasattr(page_image, 'save'):  # PPM or similar format
                    # Convert to RGB if needed
                    rgb_image = page_image.convert('RGB')
                    return np.array(rgb_image)
                    
                logger.warning(f"Unknown image type: {type(page_image)}")
                return None
                
        except Exception as e:
            logger.error(f"Error converting image to numpy: {e}")
            return None

class FeatureExtractor:
    """Utility class for extracting features from pages"""
    
    def __init__(self):
        self.text_embedder = None
        self.advanced_extractor = AdvancedFeatureExtractor()
        self._init_text_embedder()
    
    def _init_text_embedder(self):
        """Initialize text embedding model in a separate thread"""
        def load_embedder():
            try:
                from sentence_transformers import SentenceTransformer
                self.text_embedder = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Text embedder loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load text embedder: {e}")
                self.text_embedder = None
        
        # Load in background to avoid blocking UI
        threading.Thread(target=load_embedder, daemon=True).start()
    
    def get_text_embeddings(self, text: str) -> Optional[np.ndarray]:
        """Generate text embeddings for semantic similarity"""
        if not self.text_embedder or not text.strip():
            return None
        
        try:
            embeddings = self.text_embedder.encode([text])
            return embeddings[0]
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return None
    
    def get_layout_features(self, page_image, pdf_page) -> AdvancedLayoutFeatures:
        """Extract advanced layout features from page image and PDF data"""
        try:
            # Convert image to numpy array if needed
            np_image = self.advanced_extractor._convert_image_to_numpy(page_image)
            
            # Get page dimensions
            page_rect = pdf_page.rect
            page_width = page_rect.width
            page_height = page_rect.height
            
            # Extract basic text blocks
            text_blocks = self._extract_text_blocks(pdf_page)
            
            # Extract image regions
            image_regions = self._extract_image_regions(pdf_page)
            
            # Calculate white space ratio
            white_space_ratio = self._calculate_white_space_ratio(np_image)
            
            # Count text lines and get font sizes
            text_line_count = len(text_blocks)
            font_sizes = [int(block.font_size) for block in text_blocks if block.font_size > 0]
            
            # Advanced analysis
            text_distribution = self._analyze_text_distribution(text_blocks, page_width, page_height, np_image)
            font_variation_score = self._calculate_font_variation_score(font_sizes)
            text_alignment_pattern = self._detect_text_alignment_pattern(text_blocks)
            content_hierarchy_score = self._calculate_content_hierarchy_score(text_blocks)
            visual_balance_score = self._calculate_visual_balance_score(text_blocks, image_regions, page_width, page_height)
            reading_complexity_score = self._calculate_reading_complexity_score(text_blocks, text_distribution)
            
            return AdvancedLayoutFeatures(
                text_blocks=text_blocks,
                image_regions=image_regions,
                white_space_ratio=white_space_ratio,
                text_line_count=text_line_count,
                font_sizes=font_sizes,
                page_width=page_width,
                page_height=page_height,
                text_distribution=text_distribution,
                font_variation_score=font_variation_score,
                text_alignment_pattern=text_alignment_pattern,
                content_hierarchy_score=content_hierarchy_score,
                visual_balance_score=visual_balance_score,
                reading_complexity_score=reading_complexity_score
            )
        except Exception as e:
            logger.error(f"Error extracting advanced layout features: {e}")
            # Return minimal layout features
            return AdvancedLayoutFeatures(
                text_blocks=[],
                image_regions=[],
                white_space_ratio=1.0,
                text_line_count=0,
                font_sizes=[],
                page_width=0,
                page_height=0,
                text_distribution=self._get_empty_text_distribution(),
                font_variation_score=0.0,
                text_alignment_pattern="unknown",
                content_hierarchy_score=0.0,
                visual_balance_score=0.0,
                reading_complexity_score=0.0
            )
    
    def _extract_text_blocks(self, pdf_page) -> List[TextBlock]:
        """Extract text blocks from PDF page"""
        text_blocks = []
        try:
            blocks = pdf_page.get_text("dict")["blocks"]
            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            bbox = span["bbox"]
                            text_blocks.append(TextBlock(
                                x=bbox[0],
                                y=bbox[1],
                                width=bbox[2] - bbox[0],
                                height=bbox[3] - bbox[1],
                                text=span["text"],
                                font_size=span["size"]
                            ))
        except Exception as e:
            logger.error(f"Error extracting text blocks: {e}")
        
        return text_blocks
    
    def _extract_image_regions(self, pdf_page) -> List[ImageRegion]:
        """Extract image regions from PDF page"""
        image_regions = []
        try:
            image_list = pdf_page.get_images()
            for img_index, img in enumerate(image_list):
                # Get image bbox
                img_rect = pdf_page.get_image_bbox(img[7])  # img[7] is the xref
                if img_rect:
                    area = (img_rect.width) * (img_rect.height)
                    image_regions.append(ImageRegion(
                        x=img_rect.x0,
                        y=img_rect.y0,
                        width=img_rect.width,
                        height=img_rect.height,
                        area=area
                    ))
        except Exception as e:
            logger.error(f"Error extracting image regions: {e}")
        
        return image_regions
    
    def _calculate_white_space_ratio(self, page_image) -> float:
        """Calculate ratio of white space in page image"""
        try:
            if page_image is None:
                return 1.0
                
            # Ensure we have a numpy array
            if not isinstance(page_image, np.ndarray):
                return 0.5  # Default if can't process
                
            if page_image.size == 0:
                return 1.0
            
            # Convert to grayscale if needed
            if len(page_image.shape) == 3:
                gray = cv2.cvtColor(page_image, cv2.COLOR_RGB2GRAY)
            else:
                gray = page_image
            
            # Threshold to find white areas (assuming white background)
            _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
            white_pixels = np.sum(binary == 255)
            total_pixels = binary.size
            
            return white_pixels / total_pixels if total_pixels > 0 else 1.0
        except Exception as e:
            logger.error(f"Error calculating white space ratio: {e}")
            return 0.5  # Default to moderate value
    
    def detect_tables(self, page_image) -> bool:
        """Detect if page contains tables using line detection"""
        try:
            if page_image is None:
                return False
                
            # Ensure we have a numpy array
            if not isinstance(page_image, np.ndarray):
                return False
                
            if page_image.size == 0:
                return False
            
            # Convert to grayscale
            if len(page_image.shape) == 3:
                gray = cv2.cvtColor(page_image, cv2.COLOR_RGB2GRAY)
            else:
                gray = page_image
            
            # Detect horizontal and vertical lines
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
            
            horizontal_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel)
            vertical_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, vertical_kernel)
            
            # Count line pixels
            h_line_pixels = np.sum(horizontal_lines > 0)
            v_line_pixels = np.sum(vertical_lines > 0)
            
            # Threshold for table detection
            min_line_pixels = gray.size * 0.001  # 0.1% of image
            
            return h_line_pixels > min_line_pixels and v_line_pixels > min_line_pixels
        except Exception as e:
            logger.error(f"Error detecting tables: {e}")
            return False
    
    def _get_empty_text_distribution(self) -> TextDistribution:
        """Return empty text distribution for error cases"""
        return TextDistribution(
            text_density_map=np.zeros((10, 10)),
            text_clusters=[],
            vertical_distribution=np.zeros(10),
            horizontal_distribution=np.zeros(10),
            text_flow_pattern="unknown",
            margin_analysis={"top": 0, "bottom": 0, "left": 0, "right": 0},
            line_spacing_stats={"mean": 0, "std": 0, "median": 0},
            text_block_sizes=[],
            text_regularity_score=0.0
        )
    
    def _analyze_text_distribution(self, text_blocks: List[TextBlock], page_width: float, page_height: float, page_image) -> TextDistribution:
        """Perform comprehensive text distribution analysis"""
        try:
            if not text_blocks or page_width == 0 or page_height == 0:
                return self._get_empty_text_distribution()
            
            # Create high-resolution density map
            grid_size = 50
            density_map = np.zeros((grid_size, grid_size))
            
            # Calculate text density for each grid cell
            for block in text_blocks:
                if block.width > 0 and block.height > 0:
                    # Normalize coordinates
                    x_start = int((block.x / page_width) * grid_size)
                    y_start = int((block.y / page_height) * grid_size)
                    x_end = min(int(((block.x + block.width) / page_width) * grid_size), grid_size - 1)
                    y_end = min(int(((block.y + block.height) / page_height) * grid_size), grid_size - 1)
                    
                    # Add text density based on character count
                    text_weight = len(block.text.strip())
                    for y in range(max(0, y_start), min(grid_size, y_end + 1)):
                        for x in range(max(0, x_start), min(grid_size, x_end + 1)):
                            density_map[y, x] += text_weight
            
            # Analyze vertical and horizontal distributions
            vertical_dist = np.sum(density_map, axis=1)
            horizontal_dist = np.sum(density_map, axis=0)
            
            # Cluster text blocks
            text_clusters = self._cluster_text_blocks(text_blocks)
            
            # Detect text flow pattern
            text_flow_pattern = self._detect_text_flow_pattern(text_blocks, page_width, page_height)
            
            # Analyze margins
            margin_analysis = self._analyze_margins(text_blocks, page_width, page_height)
            
            # Calculate line spacing statistics
            line_spacing_stats = self._calculate_line_spacing_stats(text_blocks)
            
            # Get text block sizes
            text_block_sizes = [block.width * block.height for block in text_blocks if block.width > 0 and block.height > 0]
            
            # Calculate text regularity score
            text_regularity_score = self._calculate_text_regularity_score(text_blocks, density_map)
            
            return TextDistribution(
                text_density_map=density_map,
                text_clusters=text_clusters,
                vertical_distribution=vertical_dist,
                horizontal_distribution=horizontal_dist,
                text_flow_pattern=text_flow_pattern,
                margin_analysis=margin_analysis,
                line_spacing_stats=line_spacing_stats,
                text_block_sizes=text_block_sizes,
                text_regularity_score=text_regularity_score
            )
            
        except Exception as e:
            logger.error(f"Error analyzing text distribution: {e}")
            return self._get_empty_text_distribution()
    
    def _cluster_text_blocks(self, text_blocks: List[TextBlock]) -> List[Dict]:
        """Cluster text blocks based on spatial proximity and characteristics"""
        try:
            if len(text_blocks) < 2:
                return []
            
            # Create feature matrix for clustering
            features = []
            for block in text_blocks:
                features.append([
                    block.x, block.y, block.width, block.height,
                    block.font_size, len(block.text.strip())
                ])
            
            features = np.array(features)
            if features.shape[0] < 2:
                return []
            
            # Normalize features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Use DBSCAN for clustering (handles variable number of clusters)
            clustering = DBSCAN(eps=0.5, min_samples=2).fit(features_scaled)
            
            # Group blocks by cluster
            clusters = []
            unique_labels = set(clustering.labels_)
            for label in unique_labels:
                if label == -1:  # Noise points
                    continue
                cluster_indices = np.where(clustering.labels_ == label)[0]
                cluster_blocks = [text_blocks[i] for i in cluster_indices]
                
                # Calculate cluster statistics
                cluster_info = {
                    'blocks': cluster_blocks,
                    'size': len(cluster_blocks),
                    'avg_font_size': np.mean([b.font_size for b in cluster_blocks]),
                    'total_text_length': sum(len(b.text.strip()) for b in cluster_blocks),
                    'bounding_box': self._get_cluster_bounding_box(cluster_blocks)
                }
                clusters.append(cluster_info)
            
            return clusters
            
        except Exception as e:
            logger.error(f"Error clustering text blocks: {e}")
            return []
    
    def _get_cluster_bounding_box(self, blocks: List[TextBlock]) -> Dict[str, float]:
        """Get bounding box for a cluster of text blocks"""
        if not blocks:
            return {'x': 0, 'y': 0, 'width': 0, 'height': 0}
        
        min_x = min(block.x for block in blocks)
        min_y = min(block.y for block in blocks)
        max_x = max(block.x + block.width for block in blocks)
        max_y = max(block.y + block.height for block in blocks)
        
        return {
            'x': min_x,
            'y': min_y,
            'width': max_x - min_x,
            'height': max_y - min_y
        }
    
    def _detect_text_flow_pattern(self, text_blocks: List[TextBlock], page_width: float, page_height: float) -> str:
        """Detect the text flow pattern on the page"""
        try:
            if not text_blocks:
                return "unknown"
            
            # Analyze horizontal distribution of text blocks
            left_blocks = sum(1 for block in text_blocks if block.x < page_width * 0.3)
            center_blocks = sum(1 for block in text_blocks if page_width * 0.3 <= block.x <= page_width * 0.7)
            right_blocks = sum(1 for block in text_blocks if block.x > page_width * 0.7)
            
            total_blocks = len(text_blocks)
            
            # Analyze vertical alignment
            x_positions = [block.x for block in text_blocks]
            x_std = np.std(x_positions) if len(x_positions) > 1 else 0
            
            # Determine pattern
            if x_std < page_width * 0.05:  # Very consistent x positions
                return "single_column"
            elif left_blocks > total_blocks * 0.3 and right_blocks > total_blocks * 0.3:
                return "multi_column"
            elif center_blocks > total_blocks * 0.6:
                return "centered"
            elif x_std > page_width * 0.2:
                return "scattered"
            else:
                return "structured"
                
        except Exception as e:
            logger.error(f"Error detecting text flow pattern: {e}")
            return "unknown"
    
    def _analyze_margins(self, text_blocks: List[TextBlock], page_width: float, page_height: float) -> Dict[str, float]:
        """Analyze page margins based on text placement"""
        try:
            if not text_blocks:
                return {"top": 0, "bottom": 0, "left": 0, "right": 0}
            
            # Find text boundaries
            min_x = min(block.x for block in text_blocks)
            max_x = max(block.x + block.width for block in text_blocks)
            min_y = min(block.y for block in text_blocks)
            max_y = max(block.y + block.height for block in text_blocks)
            
            # Calculate margins as percentages
            left_margin = min_x / page_width if page_width > 0 else 0
            right_margin = (page_width - max_x) / page_width if page_width > 0 else 0
            top_margin = min_y / page_height if page_height > 0 else 0
            bottom_margin = (page_height - max_y) / page_height if page_height > 0 else 0
            
            return {
                "top": max(0, min(1, top_margin)),
                "bottom": max(0, min(1, bottom_margin)),
                "left": max(0, min(1, left_margin)),
                "right": max(0, min(1, right_margin))
            }
            
        except Exception as e:
            logger.error(f"Error analyzing margins: {e}")
            return {"top": 0, "bottom": 0, "left": 0, "right": 0}
    
    def _calculate_line_spacing_stats(self, text_blocks: List[TextBlock]) -> Dict[str, float]:
        """Calculate line spacing statistics"""
        try:
            if len(text_blocks) < 2:
                return {"mean": 0, "std": 0, "median": 0}
            
            # Sort blocks by y position
            sorted_blocks = sorted(text_blocks, key=lambda b: b.y)
            
            # Calculate spacing between consecutive blocks
            spacings = []
            for i in range(len(sorted_blocks) - 1):
                current_bottom = sorted_blocks[i].y + sorted_blocks[i].height
                next_top = sorted_blocks[i + 1].y
                spacing = max(0, next_top - current_bottom)
                spacings.append(spacing)
            
            if not spacings:
                return {"mean": 0, "std": 0, "median": 0}
            
            return {
                "mean": float(np.mean(spacings)),
                "std": float(np.std(spacings)),
                "median": float(np.median(spacings))
            }
            
        except Exception as e:
            logger.error(f"Error calculating line spacing stats: {e}")
            return {"mean": 0, "std": 0, "median": 0}
    
    def _calculate_text_regularity_score(self, text_blocks: List[TextBlock], density_map: np.ndarray) -> float:
        """Calculate how regular/structured the text layout is"""
        try:
            if not text_blocks or density_map.size == 0:
                return 0.0
            
            # Analyze spatial regularity
            x_positions = [block.x for block in text_blocks]
            y_positions = [block.y for block in text_blocks]
            
            # Calculate coefficient of variation for positions
            x_cv = np.std(x_positions) / np.mean(x_positions) if np.mean(x_positions) > 0 else 1
            y_cv = np.std(y_positions) / np.mean(y_positions) if np.mean(y_positions) > 0 else 1
            
            # Analyze density map regularity using entropy
            flat_density = density_map.flatten()
            # Normalize to probability distribution
            if np.sum(flat_density) > 0:
                prob_dist = flat_density / np.sum(flat_density)
                # Calculate entropy (lower entropy = more regular)
                entropy = -np.sum(prob_dist * np.log(prob_dist + 1e-10))
                max_entropy = np.log(len(prob_dist))
                entropy_score = 1 - (entropy / max_entropy) if max_entropy > 0 else 0
            else:
                entropy_score = 0
            
            # Combine spatial and density regularity
            spatial_regularity = 1 / (1 + x_cv + y_cv)  # Lower CV = higher regularity
            overall_regularity = (spatial_regularity + entropy_score) / 2
            
            return max(0, min(1, overall_regularity))
            
        except Exception as e:
            logger.error(f"Error calculating text regularity score: {e}")
            return 0.0
    
    def _calculate_font_variation_score(self, font_sizes: List[int]) -> float:
        """Calculate how much font sizes vary"""
        try:
            if not font_sizes:
                return 0.0
            
            if len(font_sizes) == 1:
                return 0.0  # No variation
            
            cv = np.std(font_sizes) / np.mean(font_sizes) if np.mean(font_sizes) > 0 else 0
            return min(1.0, cv)  # Cap at 1.0
            
        except Exception as e:
            logger.error(f"Error calculating font variation score: {e}")
            return 0.0
    
    def _detect_text_alignment_pattern(self, text_blocks: List[TextBlock]) -> str:
        """Detect the predominant text alignment pattern"""
        try:
            if not text_blocks:
                return "unknown"
            
            # Analyze x-positions of text blocks
            x_positions = [block.x for block in text_blocks]
            
            if len(set(x_positions)) == 1:
                return "left"  # All blocks start at same x
            
            # Check for common alignment patterns
            x_positions_rounded = [round(x, -1) for x in x_positions]  # Round to nearest 10
            position_counts = {}
            for pos in x_positions_rounded:
                position_counts[pos] = position_counts.get(pos, 0) + 1
            
            # If most blocks align to a few positions, it's structured
            max_alignment = max(position_counts.values()) if position_counts else 0
            total_blocks = len(text_blocks)
            
            if max_alignment > total_blocks * 0.7:
                return "left"
            elif len(position_counts) <= 3 and max_alignment > total_blocks * 0.4:
                return "structured"
            else:
                return "mixed"
                
        except Exception as e:
            logger.error(f"Error detecting text alignment pattern: {e}")
            return "unknown"
    
    def _calculate_content_hierarchy_score(self, text_blocks: List[TextBlock]) -> float:
        """Calculate how hierarchical the content structure is"""
        try:
            if not text_blocks:
                return 0.0
            
            font_sizes = [block.font_size for block in text_blocks if block.font_size > 0]
            if not font_sizes:
                return 0.0
            
            # Count distinct font sizes
            unique_sizes = len(set(font_sizes))
            
            # Analyze size distribution
            size_ratio = max(font_sizes) / min(font_sizes) if min(font_sizes) > 0 else 1
            
            # Higher number of distinct sizes and higher size ratio = more hierarchical
            hierarchy_score = min(1.0, (unique_sizes / 5) * (size_ratio / 3))
            
            return hierarchy_score
            
        except Exception as e:
            logger.error(f"Error calculating content hierarchy score: {e}")
            return 0.0
    
    def _calculate_visual_balance_score(self, text_blocks: List[TextBlock], image_regions: List[ImageRegion], page_width: float, page_height: float) -> float:
        """Calculate how visually balanced the page is"""
        try:
            if page_width == 0 or page_height == 0:
                return 0.0
            
            # Create weight matrix
            grid_size = 20
            weight_map = np.zeros((grid_size, grid_size))
            
            # Add text weights
            for block in text_blocks:
                if block.width > 0 and block.height > 0:
                    x_center = int((block.x + block.width/2) / page_width * grid_size)
                    y_center = int((block.y + block.height/2) / page_height * grid_size)
                    x_center = max(0, min(grid_size-1, x_center))
                    y_center = max(0, min(grid_size-1, y_center))
                    weight_map[y_center, x_center] += len(block.text.strip())
            
            # Add image weights
            for img in image_regions:
                if img.area > 0:
                    x_center = int((img.x + img.width/2) / page_width * grid_size)
                    y_center = int((img.y + img.height/2) / page_height * grid_size)
                    x_center = max(0, min(grid_size-1, x_center))
                    y_center = max(0, min(grid_size-1, y_center))
                    weight_map[y_center, x_center] += img.area / 1000  # Scale image weight
            
            # Calculate balance (center of mass vs geometric center)
            if np.sum(weight_map) == 0:
                return 0.0
            
            # Find center of mass
            y_indices, x_indices = np.mgrid[0:grid_size, 0:grid_size]
            total_weight = np.sum(weight_map)
            center_of_mass_x = np.sum(x_indices * weight_map) / total_weight
            center_of_mass_y = np.sum(y_indices * weight_map) / total_weight
            
            # Geometric center
            geometric_center_x = grid_size / 2
            geometric_center_y = grid_size / 2
            
            # Distance from center (normalized)
            distance = np.sqrt((center_of_mass_x - geometric_center_x)**2 + 
                             (center_of_mass_y - geometric_center_y)**2)
            max_distance = np.sqrt((grid_size/2)**2 + (grid_size/2)**2)
            
            balance_score = 1 - (distance / max_distance)
            return max(0, min(1, balance_score))
            
        except Exception as e:
            logger.error(f"Error calculating visual balance score: {e}")
            return 0.0
    
    def _calculate_reading_complexity_score(self, text_blocks: List[TextBlock], text_distribution: TextDistribution) -> float:
        """Calculate how complex the reading pattern is"""
        try:
            if not text_blocks:
                return 0.0
            
            # Factors that increase reading complexity:
            complexity_factors = []
            
            # 1. Number of distinct text clusters
            num_clusters = len(text_distribution.text_clusters)
            cluster_complexity = min(1.0, num_clusters / 5)
            complexity_factors.append(cluster_complexity)
            
            # 2. Text flow pattern complexity
            flow_complexity_map = {
                "single_column": 0.1,
                "multi_column": 0.4,
                "centered": 0.3,
                "structured": 0.2,
                "scattered": 0.9,
                "unknown": 0.5
            }
            flow_complexity = flow_complexity_map.get(text_distribution.text_flow_pattern, 0.5)
            complexity_factors.append(flow_complexity)
            
            # 3. Font variation (more fonts = more complex)
            font_sizes = [block.font_size for block in text_blocks if block.font_size > 0]
            if font_sizes:
                font_variation = len(set(font_sizes)) / 10  # Normalize by typical max variety
                complexity_factors.append(min(1.0, font_variation))
            
            # 4. Spatial irregularity
            irregularity = 1 - text_distribution.text_regularity_score
            complexity_factors.append(irregularity)
            
            # Combine factors
            overall_complexity = np.mean(complexity_factors)
            return max(0, min(1, overall_complexity))
            
        except Exception as e:
            logger.error(f"Error calculating reading complexity score: {e}")
            return 0.0

class PageAnalyzer:
    """Analyzes individual pages for content characteristics"""
    
    def __init__(self):
        self.feature_extractor = FeatureExtractor()
    
    def extract_text_features(self, pdf_page) -> Tuple[str, Optional[np.ndarray]]:
        """Extract text content and embeddings"""
        try:
            text_content = pdf_page.get_text()
            text_embedding = self.feature_extractor.get_text_embeddings(text_content)
            return text_content, text_embedding
        except Exception as e:
            logger.error(f"Error extracting text features: {e}")
            return "", None
    
    def extract_visual_features(self, page_image, pdf_page) -> AdvancedLayoutFeatures:
        """Extract advanced visual/layout features"""
        return self.feature_extractor.get_layout_features(page_image, pdf_page)
    
    def detect_content_type(self, page_features: PageFeatures) -> ContentType:
        """Classify page content type based on features with better image content distinction"""
        try:
            # Check for blank pages first
            if page_features.is_blank:
                return ContentType.BLANK
            
            # Check for tables
            if page_features.has_tables:
                return ContentType.TABLE
            
            # Check text density and image presence
            text_density = page_features.text_density
            white_space = page_features.layout_features.white_space_ratio
            has_images = page_features.has_images
            
            # Title pages: low text density, high white space, large fonts
            if (text_density < 0.1 and white_space > 0.7 and 
                page_features.layout_features.font_sizes and 
                max(page_features.layout_features.font_sizes) > 16):
                return ContentType.TITLE
            
            # Image-heavy pages (prioritize this detection)
            if has_images:
                image_count = len(page_features.layout_features.image_regions)
                total_image_area = sum(img.area for img in page_features.layout_features.image_regions)
                
                # If there are significant images and low text density, classify by image type
                if text_density < 0.2 and (image_count > 0 or total_image_area > 1000):
                    if image_count == 1 and text_density < 0.05:
                        # Single image with very little text (like your banana/phone pages)
                        return ContentType.SINGLE_IMAGE
                    elif text_density < 0.1:
                        # Multiple images or image with little text
                        return ContentType.IMAGE_HEAVY
                    else:
                        # Images with more text
                        return ContentType.MIXED
            
            # Text-heavy pages
            if text_density > 0.3:
                return ContentType.TEXT_HEAVY
            
            # Default to mixed content
            return ContentType.MIXED
            
        except Exception as e:
            logger.error(f"Error detecting content type: {e}")
            return ContentType.MIXED
    
    def calculate_text_density(self, text_content: str, layout_features: AdvancedLayoutFeatures) -> float:
        """Calculate text density (characters per unit area)"""
        try:
            if not text_content or layout_features.page_width == 0 or layout_features.page_height == 0:
                return 0.0
            
            char_count = len(text_content.strip())
            page_area = layout_features.page_width * layout_features.page_height
            
            return char_count / page_area
        except Exception as e:
            logger.error(f"Error calculating text density: {e}")
            return 0.0

class SmartSelector:
    """Main smart selection engine"""
    
    def __init__(self):
        self.page_analyzer = PageAnalyzer()
        self.document_features: Dict[int, PageFeatures] = {}
        self.analysis_complete = False
    
    def analyze_document(self, pdf_path: str, page_images: List[np.ndarray]) -> bool:
        """Analyze entire document and extract features for all pages"""
        try:
            logger.info(f"Starting document analysis for {pdf_path}")
            self.document_features.clear()
            
            # Open PDF
            pdf_doc = fitz.open(pdf_path)
            total_pages = len(pdf_doc)
            
            logger.info(f"Document has {total_pages} pages, page_images has {len(page_images)} images")
            
            for page_num in range(total_pages):
                try:
                    logger.info(f"Analyzing page {page_num + 1}/{total_pages}")
                    pdf_page = pdf_doc[page_num]
                    page_image = page_images[page_num] if page_num < len(page_images) else None
                    
                    # Extract features
                    features = self._extract_page_features(pdf_page, page_image, page_num)
                    self.document_features[page_num] = features
                    
                    logger.info(f"Successfully analyzed page {page_num + 1}")
                    
                except Exception as e:
                    logger.error(f"Error analyzing page {page_num + 1}: {e}")
                    # Create minimal features for failed pages
                    try:
                        empty_distribution = self.page_analyzer.feature_extractor._get_empty_text_distribution()
                        empty_layout = AdvancedLayoutFeatures(
                            text_blocks=[], image_regions=[], white_space_ratio=1.0, text_line_count=0, 
                            font_sizes=[], page_width=0, page_height=0, text_distribution=empty_distribution,
                            font_variation_score=0.0, text_alignment_pattern="unknown", content_hierarchy_score=0.0,
                            visual_balance_score=0.0, reading_complexity_score=0.0
                        )
                        minimal_features = PageFeatures(
                            text_embedding=None, text_density=0.0, layout_features=empty_layout,
                            content_type=ContentType.MIXED, has_tables=False, has_images=False,
                            is_blank=True, text_content="", structure_fingerprint=np.zeros(50),
                            visual_complexity_score=0.0, information_density_score=0.0
                        )
                        self.document_features[page_num] = minimal_features
                        logger.info(f"Added minimal features for page {page_num + 1}")
                    except Exception as e2:
                        logger.error(f"Failed to create minimal features for page {page_num + 1}: {e2}")
                        continue
            
            pdf_doc.close()
            self.analysis_complete = True
            logger.info(f"Document analysis complete. Successfully analyzed {len(self.document_features)} out of {total_pages} pages.")
            return True
            
        except Exception as e:
            logger.error(f"Error during document analysis: {e}")
            return False
    
    def _extract_page_features(self, pdf_page, page_image: Optional[np.ndarray], page_num: int) -> PageFeatures:
        """Extract complete feature set for a single page"""
        try:
            # Extract text features
            text_content, text_embedding = self.page_analyzer.extract_text_features(pdf_page)
            
            # Extract visual features
            layout_features = self.page_analyzer.extract_visual_features(page_image, pdf_page)
            
            # Calculate derived features
            text_density = self.page_analyzer.calculate_text_density(text_content, layout_features)
            
            # Detect content characteristics
            is_blank = text_density < 0.001 and layout_features.white_space_ratio > 0.95
            has_images = len(layout_features.image_regions) > 0
            has_tables = self.page_analyzer.feature_extractor.detect_tables(page_image) if page_image is not None else False
            
            # Create advanced features for similarity matching
            structure_fingerprint = self._create_structure_fingerprint(layout_features)
            visual_complexity_score = self._calculate_visual_complexity_score(layout_features)
            information_density_score = self._calculate_information_density_score(layout_features, text_density)
            
            # Create page features object
            page_features = PageFeatures(
                text_embedding=text_embedding,
                text_density=text_density,
                layout_features=layout_features,
                content_type=ContentType.MIXED,  # Will be set below
                has_tables=has_tables,
                has_images=has_images,
                is_blank=is_blank,
                text_content=text_content,
                structure_fingerprint=structure_fingerprint,
                visual_complexity_score=visual_complexity_score,
                information_density_score=information_density_score
            )
            
            # Detect content type
            page_features.content_type = self.page_analyzer.detect_content_type(page_features)
            
            return page_features
            
        except Exception as e:
            logger.error(f"Error extracting features for page {page_num}: {e}")
            # Return minimal features
            empty_distribution = self.page_analyzer.feature_extractor._get_empty_text_distribution()
            empty_layout = AdvancedLayoutFeatures(
                text_blocks=[], image_regions=[], white_space_ratio=1.0, text_line_count=0, 
                font_sizes=[], page_width=0, page_height=0, text_distribution=empty_distribution,
                font_variation_score=0.0, text_alignment_pattern="unknown", content_hierarchy_score=0.0,
                visual_balance_score=0.0, reading_complexity_score=0.0
            )
            return PageFeatures(
                text_embedding=None,
                text_density=0.0,
                layout_features=empty_layout,
                content_type=ContentType.MIXED,
                has_tables=False,
                has_images=False,
                is_blank=True,
                text_content="",
                structure_fingerprint=np.zeros(50),
                visual_complexity_score=0.0,
                information_density_score=0.0
            )
    
    def find_similar_pages(self, example_pages: List[int], similarity_threshold: float = 0.7) -> SelectionResult:
        """Find pages similar to the provided examples"""
        if not self.analysis_complete or not example_pages:
            return SelectionResult([], [], "Analysis not complete or no examples provided", [], 0, 0.0)
        
        import time
        start_time = time.time()
        
        try:
            # Get example features
            example_features = [self.document_features[page] for page in example_pages if page in self.document_features]
            if not example_features:
                return SelectionResult([], [], "No valid example pages found", [], 0, 0.0)
            
            # Log example page details for debugging
            for i, (page_idx, example) in enumerate(zip(example_pages, example_features)):
                logger.info(f"Example page {page_idx + 1}: "
                          f"Content: {example.content_type.value}, "
                          f"Text density: {example.text_density:.4f}, "
                          f"Lines: {example.layout_features.text_line_count}, "
                          f"White space: {example.layout_features.white_space_ratio:.3f}")
            
            similar_pages = []
            confidence_scores = []
            features_found = set()
            
            # Compare all pages against examples
            for page_num, page_features in self.document_features.items():
                if page_num in example_pages:
                    continue  # Skip example pages
                
                # Calculate similarity
                similarity = self._calculate_page_similarity(example_features, page_features)
                
                # Debug logging with more detail
                logger.info(f"Page {page_num + 1} similarity: {similarity:.3f} (threshold: {similarity_threshold:.3f}) - "
                          f"Content: {page_features.content_type.value}, "
                          f"Text density: {page_features.text_density:.4f}, "
                          f"Lines: {page_features.layout_features.text_line_count}, "
                          f"White space: {page_features.layout_features.white_space_ratio:.3f}")
                
                if similarity >= similarity_threshold:
                    similar_pages.append(page_num)
                    confidence_scores.append(similarity)
                    logger.info(f"Page {page_num + 1} SELECTED (similarity: {similarity:.3f})")
                    
                    # Track common features
                    if page_features.content_type:
                        features_found.add(page_features.content_type.value)
            
            processing_time = time.time() - start_time
            
            # Sort by confidence
            if similar_pages:
                paired = list(zip(similar_pages, confidence_scores))
                paired.sort(key=lambda x: x[1], reverse=True)
                similar_pages, confidence_scores = zip(*paired)
                similar_pages = list(similar_pages)
                confidence_scores = list(confidence_scores)
            
            reasoning = f"Found {len(similar_pages)} pages similar to examples {example_pages}"
            if features_found:
                reasoning += f" with features: {', '.join(features_found)}"
            
            return SelectionResult(
                selected_pages=similar_pages,
                confidence_scores=confidence_scores,
                reasoning=reasoning,
                features_found=list(features_found),
                total_pages_analyzed=len(self.document_features),
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Error finding similar pages: {e}")
            return SelectionResult([], [], f"Error during similarity search: {str(e)}", [], 0, 0.0)
    
    def find_pages_by_content_type(self, content_type: ContentType) -> SelectionResult:
        """Find pages of a specific content type"""
        if not self.analysis_complete:
            return SelectionResult([], [], "Analysis not complete", [], 0, 0.0)
        
        import time
        start_time = time.time()
        
        try:
            matching_pages = []
            confidence_scores = []
            
            for page_num, page_features in self.document_features.items():
                if page_features.content_type == content_type:
                    matching_pages.append(page_num)
                    # Confidence based on feature strength
                    confidence = self._calculate_content_type_confidence(page_features, content_type)
                    confidence_scores.append(confidence)
            
            processing_time = time.time() - start_time
            
            reasoning = f"Found {len(matching_pages)} pages of type '{content_type.value}'"
            
            return SelectionResult(
                selected_pages=matching_pages,
                confidence_scores=confidence_scores,
                reasoning=reasoning,
                features_found=[content_type.value],
                total_pages_analyzed=len(self.document_features),
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Error finding pages by content type: {e}")
            return SelectionResult([], [], f"Error during content type search: {str(e)}", [], 0, 0.0)
    
    def _create_structure_fingerprint(self, layout_features: AdvancedLayoutFeatures) -> np.ndarray:
        """Create a unique structural fingerprint for the page"""
        try:
            fingerprint = []
            
            # Text distribution signature (flatten density map)
            if layout_features.text_distribution.text_density_map.size > 0:
                # Downsample to fixed size for consistency
                density_resized = cv2.resize(layout_features.text_distribution.text_density_map, (10, 10))
                fingerprint.extend(density_resized.flatten())
            else:
                fingerprint.extend([0] * 100)  # 10x10 = 100 zeros
            
            # Spatial distribution features
            fingerprint.extend([
                layout_features.white_space_ratio,
                layout_features.text_line_count / 100,  # Normalize
                len(layout_features.font_sizes) / 10,   # Normalize
                layout_features.font_variation_score,
                layout_features.content_hierarchy_score,
                layout_features.visual_balance_score,
                layout_features.reading_complexity_score
            ])
            
            # Margin signature
            margins = layout_features.text_distribution.margin_analysis
            fingerprint.extend([
                margins.get('top', 0),
                margins.get('bottom', 0),
                margins.get('left', 0),
                margins.get('right', 0)
            ])
            
            # Line spacing signature
            spacing_stats = layout_features.text_distribution.line_spacing_stats
            fingerprint.extend([
                spacing_stats.get('mean', 0) / 100,  # Normalize
                spacing_stats.get('std', 0) / 100,   # Normalize
                spacing_stats.get('median', 0) / 100  # Normalize
            ])
            
            # Text flow pattern as numeric
            flow_mapping = {
                "single_column": 0.1, "multi_column": 0.3, "centered": 0.5,
                "structured": 0.7, "scattered": 0.9, "unknown": 0.5
            }
            flow_score = flow_mapping.get(layout_features.text_distribution.text_flow_pattern, 0.5)
            fingerprint.append(flow_score)
            
            # Alignment pattern as numeric
            align_mapping = {
                "left": 0.1, "center": 0.3, "right": 0.5, "structured": 0.7, "mixed": 0.9, "unknown": 0.5
            }
            align_score = align_mapping.get(layout_features.text_alignment_pattern, 0.5)
            fingerprint.append(align_score)
            
            # Pad or truncate to fixed size
            target_size = 50
            if len(fingerprint) > target_size:
                fingerprint = fingerprint[:target_size]
            else:
                fingerprint.extend([0] * (target_size - len(fingerprint)))
            
            return np.array(fingerprint, dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Error creating structure fingerprint: {e}")
            return np.zeros(50, dtype=np.float32)
    
    def _calculate_visual_complexity_score(self, layout_features: AdvancedLayoutFeatures) -> float:
        """Calculate overall visual complexity of the page"""
        try:
            complexity_factors = []
            
            # Font variation contributes to complexity
            complexity_factors.append(layout_features.font_variation_score)
            
            # Reading complexity
            complexity_factors.append(layout_features.reading_complexity_score)
            
            # Number of text clusters
            num_clusters = len(layout_features.text_distribution.text_clusters)
            cluster_complexity = min(1.0, num_clusters / 5)
            complexity_factors.append(cluster_complexity)
            
            # Visual balance (imbalanced = more complex)
            balance_complexity = 1 - layout_features.visual_balance_score
            complexity_factors.append(balance_complexity)
            
            # Text regularity (irregular = more complex)
            regularity_complexity = 1 - layout_features.text_distribution.text_regularity_score
            complexity_factors.append(regularity_complexity)
            
            # White space distribution (very high or low = complex)
            ws_complexity = abs(layout_features.white_space_ratio - 0.5) * 2  # Peak at extremes
            complexity_factors.append(ws_complexity)
            
            return np.mean(complexity_factors)
            
        except Exception as e:
            logger.error(f"Error calculating visual complexity score: {e}")
            return 0.0
    
    def _calculate_information_density_score(self, layout_features: AdvancedLayoutFeatures, text_density: float) -> float:
        """Calculate information density score"""
        try:
            density_factors = []
            
            # Basic text density
            density_factors.append(min(1.0, text_density * 10))  # Scale up
            
            # Number of text blocks per unit area
            if layout_features.page_width > 0 and layout_features.page_height > 0:
                page_area = layout_features.page_width * layout_features.page_height
                blocks_per_area = len(layout_features.text_blocks) / (page_area / 100000)  # Normalize
                density_factors.append(min(1.0, blocks_per_area))
            
            # Font size variation (more variation = more information layers)
            density_factors.append(layout_features.font_variation_score)
            
            # Content hierarchy (hierarchical content = denser information)
            density_factors.append(layout_features.content_hierarchy_score)
            
            # Image presence adds to information density
            image_density = min(1.0, len(layout_features.image_regions) / 5)
            density_factors.append(image_density)
            
            return np.mean(density_factors)
            
        except Exception as e:
            logger.error(f"Error calculating information density score: {e}")
            return 0.0
    
    def _calculate_page_similarity(self, example_features: List[PageFeatures], page_features: PageFeatures) -> float:
        """Calculate similarity between page and examples with strict visual content analysis"""
        try:
            similarities = []
            
            for example in example_features:
                # Start with base similarity of 0
                total_score = 0.0
                max_possible_score = 0.0
                
                # 1. Text content similarity (30% weight) - but only if both have meaningful text
                text_weight = 0.3
                if (example.text_embedding is not None and page_features.text_embedding is not None):
                    text_sim = self._cosine_similarity(example.text_embedding, page_features.text_embedding)
                    # Scale text similarity and make it more strict
                    text_sim = max(0, (text_sim + 1) / 2)  # Convert from [-1,1] to [0,1]
                    
                    # If text similarity is very low, heavily penalize
                    if text_sim < 0.3:
                        text_sim = text_sim * 0.5  # Cut low text similarity in half
                    
                    total_score += text_sim * text_weight
                    max_possible_score += text_weight
                
                # 2. Visual content similarity (40% weight) - STRICT image comparison
                visual_weight = 0.4
                visual_sim = self._calculate_strict_visual_similarity(example, page_features)
                total_score += visual_sim * visual_weight
                max_possible_score += visual_weight
                
                # 3. Layout structure similarity (20% weight)
                layout_weight = 0.2
                layout_sim = self._calculate_simple_layout_similarity(example.layout_features, page_features.layout_features)
                total_score += layout_sim * layout_weight
                max_possible_score += layout_weight
                
                # 4. Content type matching (10% weight) - but make it more strict
                content_weight = 0.1
                if example.content_type == page_features.content_type:
                    content_sim = 1.0
                else:
                    content_sim = 0.0  # Zero score for different content types
                
                total_score += content_sim * content_weight
                max_possible_score += content_weight
                
                # Calculate final similarity as percentage of max possible
                if max_possible_score > 0:
                    final_similarity = total_score / max_possible_score
                    
                    # Apply strict penalty for mismatched content
                    if example.content_type != page_features.content_type:
                        final_similarity = final_similarity * 0.5  # Cut score in half for different content types
                    
                    similarities.append(final_similarity)
                else:
                    similarities.append(0.0)
            
            # Return maximum similarity to any example
            return max(similarities) if similarities else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating page similarity: {e}")
            return 0.0
    
    def _calculate_strict_visual_similarity(self, example_features: PageFeatures, page_features: PageFeatures) -> float:
        """Calculate visual similarity focusing on layout structure and content patterns"""
        try:
            # Get layout features for easier access
            example_layout = example_features.layout_features
            page_layout = page_features.layout_features
            
            similarity_components = []
            
            # 1. Text density similarity (critical for form vs blank page distinction)
            example_density = example_features.text_density
            page_density = page_features.text_density
            
            # If one is very dense and other is sparse, they're very different
            if (example_density > 0.1 and page_density < 0.05) or (page_density > 0.1 and example_density < 0.05):
                return 0.1  # Form page vs mostly blank page
            
            # For similar density ranges, calculate similarity
            if max(example_density, page_density) > 0:
                density_ratio = min(example_density, page_density) / max(example_density, page_density)
                similarity_components.append(density_ratio)
            
            # 2. Text line count similarity (forms have many lines, blank pages have few)
            example_lines = example_layout.text_line_count
            page_lines = page_layout.text_line_count
            
            if max(example_lines, page_lines) > 0:
                line_ratio = min(example_lines, page_lines) / max(example_lines, page_lines)
                similarity_components.append(line_ratio)
            elif example_lines == page_lines == 0:
                similarity_components.append(1.0)  # Both have no text lines
            
            # 3. White space ratio similarity
            ws_diff = abs(example_layout.white_space_ratio - page_layout.white_space_ratio)
            ws_sim = max(0, 1 - (ws_diff * 2))  # Penalty for white space differences
            similarity_components.append(ws_sim)
            
            # 4. Font size pattern similarity (forms typically have varied font sizes)
            example_fonts = example_layout.font_sizes
            page_fonts = page_layout.font_sizes
            
            if example_fonts and page_fonts:
                # Compare font diversity
                example_font_variety = len(set(example_fonts))
                page_font_variety = len(set(page_fonts))
                
                if max(example_font_variety, page_font_variety) > 0:
                    font_variety_ratio = min(example_font_variety, page_font_variety) / max(example_font_variety, page_font_variety)
                    similarity_components.append(font_variety_ratio)
                
                # Compare average font sizes
                example_avg_font = sum(example_fonts) / len(example_fonts)
                page_avg_font = sum(page_fonts) / len(page_fonts)
                
                if max(example_avg_font, page_avg_font) > 0:
                    font_size_ratio = min(example_avg_font, page_avg_font) / max(example_avg_font, page_avg_font)
                    similarity_components.append(font_size_ratio)
            
            # 5. Page dimensions similarity
            if (example_layout.page_width > 0 and example_layout.page_height > 0 and 
                page_layout.page_width > 0 and page_layout.page_height > 0):
                
                width_ratio = min(example_layout.page_width, page_layout.page_width) / max(example_layout.page_width, page_layout.page_width)
                height_ratio = min(example_layout.page_height, page_layout.page_height) / max(example_layout.page_height, page_layout.page_height)
                size_sim = (width_ratio + height_ratio) / 2
                similarity_components.append(size_sim)
            
            # 6. Image comparison (if relevant)
            example_has_images = example_features.has_images
            page_has_images = page_features.has_images
            
            if example_has_images == page_has_images:
                if example_has_images:  # Both have images
                    example_img_count = len(example_layout.image_regions)
                    page_img_count = len(page_layout.image_regions)
                    if max(example_img_count, page_img_count) > 0:
                        img_ratio = min(example_img_count, page_img_count) / max(example_img_count, page_img_count)
                        similarity_components.append(img_ratio)
                else:  # Both have no images
                    similarity_components.append(1.0)
            else:
                similarity_components.append(0.2)  # Different image presence
            
            # 7. Content type matching
            if example_features.content_type == page_features.content_type:
                similarity_components.append(1.0)
            else:
                similarity_components.append(0.2)
            
            # Calculate final similarity
            if similarity_components:
                avg_sim = sum(similarity_components) / len(similarity_components)
                
                # Apply stricter requirements: all components should be reasonably good
                min_component = min(similarity_components)
                if min_component < 0.4:
                    avg_sim = avg_sim * 0.7  # Penalize if any component is very low
                
                return avg_sim
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Error calculating strict visual similarity: {e}")
            return 0.0
    
    def _calculate_simple_layout_similarity(self, layout1: AdvancedLayoutFeatures, layout2: AdvancedLayoutFeatures) -> float:
        """Calculate simple layout similarity that works well for identical pages"""
        try:
            similarity_components = []
            
            # 1. White space ratio (should be very similar for identical pages)
            ws_diff = abs(layout1.white_space_ratio - layout2.white_space_ratio)
            ws_sim = max(0, 1 - (ws_diff * 2))  # Scale so small differences don't kill the score
            similarity_components.append(ws_sim)
            
            # 2. Text line count similarity (normalize by larger value to handle scale differences)
            if max(layout1.text_line_count, layout2.text_line_count) > 0:
                line_ratio = min(layout1.text_line_count, layout2.text_line_count) / max(layout1.text_line_count, layout2.text_line_count)
                similarity_components.append(line_ratio)
            
            # 3. Page dimensions similarity
            if layout1.page_width > 0 and layout1.page_height > 0 and layout2.page_width > 0 and layout2.page_height > 0:
                width_ratio = min(layout1.page_width, layout2.page_width) / max(layout1.page_width, layout2.page_width)
                height_ratio = min(layout1.page_height, layout2.page_height) / max(layout1.page_height, layout2.page_height)
                size_sim = (width_ratio + height_ratio) / 2
                similarity_components.append(size_sim)
            
            # 4. Has similar number of images
            img_count1 = len(layout1.image_regions)
            img_count2 = len(layout2.image_regions)
            if max(img_count1, img_count2) > 0:
                img_ratio = min(img_count1, img_count2) / max(img_count1, img_count2)
                similarity_components.append(img_ratio)
            elif img_count1 == img_count2 == 0:
                similarity_components.append(1.0)  # Both have no images
            
            # Return average of components
            return sum(similarity_components) / len(similarity_components) if similarity_components else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating simple layout similarity: {e}")
            return 0.0
    
    def _calculate_basic_feature_similarity(self, features1: PageFeatures, features2: PageFeatures) -> float:
        """Calculate basic feature similarity"""
        try:
            matches = 0
            total_checks = 0
            
            # Boolean feature comparisons
            if features1.has_tables == features2.has_tables:
                matches += 1
            total_checks += 1
            
            if features1.has_images == features2.has_images:
                matches += 1
            total_checks += 1
            
            if features1.is_blank == features2.is_blank:
                matches += 1
            total_checks += 1
            
            # Text density similarity (scale the difference)
            density_diff = abs(features1.text_density - features2.text_density)
            density_sim = max(0, 1 - (density_diff * 100))  # Scale density difference
            matches += density_sim
            total_checks += 1
            
            return matches / total_checks if total_checks > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating basic feature similarity: {e}")
            return 0.0
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            return cosine_similarity([vec1], [vec2])[0][0]
        except Exception:
            # Fallback to manual calculation
            try:
                dot_product = np.dot(vec1, vec2)
                norm1 = np.linalg.norm(vec1)
                norm2 = np.linalg.norm(vec2)
                if norm1 == 0 or norm2 == 0:
                    return 0.0
                return dot_product / (norm1 * norm2)
            except Exception as e:
                logger.error(f"Error calculating cosine similarity: {e}")
                return 0.0
    
    def _compare_advanced_layout_features(self, layout1: AdvancedLayoutFeatures, layout2: AdvancedLayoutFeatures) -> float:
        """Compare advanced layout features between two pages"""
        try:
            similarity_factors = []
            
            # 1. Text distribution pattern similarity
            if (layout1.text_distribution.text_density_map.size > 0 and 
                layout2.text_distribution.text_density_map.size > 0):
                # Resize both to same size for comparison
                map1_resized = cv2.resize(layout1.text_distribution.text_density_map, (20, 20))
                map2_resized = cv2.resize(layout2.text_distribution.text_density_map, (20, 20))
                
                # Normalize
                if np.max(map1_resized) > 0:
                    map1_norm = map1_resized / np.max(map1_resized)
                else:
                    map1_norm = map1_resized
                    
                if np.max(map2_resized) > 0:
                    map2_norm = map2_resized / np.max(map2_resized)
                else:
                    map2_norm = map2_resized
                
                # Calculate correlation
                if np.std(map1_norm) > 0 and np.std(map2_norm) > 0:
                    correlation = np.corrcoef(map1_norm.flatten(), map2_norm.flatten())[0, 1]
                    if not np.isnan(correlation):
                        similarity_factors.append(abs(correlation))
            
            # 2. Text flow pattern similarity
            flow_sim = 1.0 if layout1.text_distribution.text_flow_pattern == layout2.text_distribution.text_flow_pattern else 0.3
            similarity_factors.append(flow_sim)
            
            # 3. Text alignment pattern similarity
            align_sim = 1.0 if layout1.text_alignment_pattern == layout2.text_alignment_pattern else 0.3
            similarity_factors.append(align_sim)
            
            # 4. Margin similarity
            margin1 = layout1.text_distribution.margin_analysis
            margin2 = layout2.text_distribution.margin_analysis
            margin_diffs = [
                abs(margin1.get('top', 0) - margin2.get('top', 0)),
                abs(margin1.get('bottom', 0) - margin2.get('bottom', 0)),
                abs(margin1.get('left', 0) - margin2.get('left', 0)),
                abs(margin1.get('right', 0) - margin2.get('right', 0))
            ]
            margin_sim = 1 - np.mean(margin_diffs)
            similarity_factors.append(max(0, margin_sim))
            
            # 5. Line spacing similarity
            spacing1 = layout1.text_distribution.line_spacing_stats
            spacing2 = layout2.text_distribution.line_spacing_stats
            if spacing1.get('mean', 0) > 0 and spacing2.get('mean', 0) > 0:
                spacing_diff = abs(spacing1['mean'] - spacing2['mean']) / max(spacing1['mean'], spacing2['mean'])
                spacing_sim = 1 - spacing_diff
                similarity_factors.append(max(0, spacing_sim))
            
            # 6. Font variation similarity
            font_var_diff = abs(layout1.font_variation_score - layout2.font_variation_score)
            font_var_sim = 1 - font_var_diff
            similarity_factors.append(max(0, font_var_sim))
            
            # 7. Content hierarchy similarity
            hierarchy_diff = abs(layout1.content_hierarchy_score - layout2.content_hierarchy_score)
            hierarchy_sim = 1 - hierarchy_diff
            similarity_factors.append(max(0, hierarchy_sim))
            
            # 8. Visual balance similarity
            balance_diff = abs(layout1.visual_balance_score - layout2.visual_balance_score)
            balance_sim = 1 - balance_diff
            similarity_factors.append(max(0, balance_sim))
            
            # 9. Reading complexity similarity
            complexity_diff = abs(layout1.reading_complexity_score - layout2.reading_complexity_score)
            complexity_sim = 1 - complexity_diff
            similarity_factors.append(max(0, complexity_sim))
            
            # 10. Text regularity similarity
            regularity_diff = abs(layout1.text_distribution.text_regularity_score - layout2.text_distribution.text_regularity_score)
            regularity_sim = 1 - regularity_diff
            similarity_factors.append(max(0, regularity_sim))
            
            # 11. White space ratio similarity
            ws_diff = abs(layout1.white_space_ratio - layout2.white_space_ratio)
            ws_sim = 1 - ws_diff
            similarity_factors.append(max(0, ws_sim))
            
            # 12. Text line count similarity (normalized)
            if max(layout1.text_line_count, layout2.text_line_count) > 0:
                line_diff = abs(layout1.text_line_count - layout2.text_line_count) / max(layout1.text_line_count, layout2.text_line_count)
                line_sim = 1 - line_diff
                similarity_factors.append(max(0, line_sim))
            
            return np.mean(similarity_factors) if similarity_factors else 0.0
            
        except Exception as e:
            logger.error(f"Error comparing advanced layout features: {e}")
            return 0.0

    def _compare_layout_features(self, layout1: AdvancedLayoutFeatures, layout2: AdvancedLayoutFeatures) -> float:
        """Compare layout features between two pages"""
        try:
            similarity = 0.0
            
            # White space ratio similarity
            ws_diff = abs(layout1.white_space_ratio - layout2.white_space_ratio)
            ws_sim = max(0, 1 - ws_diff)
            similarity += 0.4 * ws_sim
            
            # Text line count similarity
            max_lines = max(layout1.text_line_count, layout2.text_line_count, 1)
            line_diff = abs(layout1.text_line_count - layout2.text_line_count)
            line_sim = max(0, 1 - line_diff / max_lines)
            similarity += 0.3 * line_sim
            
            # Font size similarity
            if layout1.font_sizes and layout2.font_sizes:
                avg_font1 = sum(layout1.font_sizes) / len(layout1.font_sizes)
                avg_font2 = sum(layout2.font_sizes) / len(layout2.font_sizes)
                font_diff = abs(avg_font1 - avg_font2)
                font_sim = max(0, 1 - font_diff / 20)  # Normalize by typical font size range
                similarity += 0.3 * font_sim
            
            return similarity
            
        except Exception as e:
            logger.error(f"Error comparing layout features: {e}")
            return 0.0
    
    def _compare_features(self, features1: PageFeatures, features2: PageFeatures) -> float:
        """Compare boolean features between two pages"""
        try:
            similarity = 0.0
            
            # Boolean feature comparisons
            if features1.has_tables == features2.has_tables:
                similarity += 0.33
            if features1.has_images == features2.has_images:
                similarity += 0.33
            if features1.is_blank == features2.is_blank:
                similarity += 0.34
            
            return similarity
            
        except Exception as e:
            logger.error(f"Error comparing features: {e}")
            return 0.0
    
    def _calculate_content_type_confidence(self, page_features: PageFeatures, content_type: ContentType) -> float:
        """Calculate confidence score for content type classification"""
        try:
            if content_type == ContentType.BLANK:
                return 1.0 if page_features.is_blank else 0.8
            elif content_type == ContentType.TABLE:
                return 0.95 if page_features.has_tables else 0.7
            elif content_type == ContentType.TEXT_HEAVY:
                return min(1.0, page_features.text_density * 3)  # Scale text density
            elif content_type == ContentType.IMAGE_HEAVY:
                return 0.9 if page_features.has_images else 0.6
            elif content_type == ContentType.TITLE:
                # Check for title characteristics
                layout = page_features.layout_features
                if layout.font_sizes and max(layout.font_sizes) > 16 and layout.white_space_ratio > 0.6:
                    return 0.9
                return 0.7
            else:
                return 0.8  # Default confidence for mixed content
                
        except Exception as e:
            logger.error(f"Error calculating content type confidence: {e}")
            return 0.5 