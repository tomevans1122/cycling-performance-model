#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPX TOOL MODULE
---------------
This module handles all terrain data processing.
1. Parsing: Reads raw GPX files (XML).
2. Cleaning: Removes GPS jitter using smoothing algorithms.
3. Enrichment: Calculates gradients, bearings, and distances.
4. Segmentation: Breaks the course into logical chunks (Climb, Flat, Descent) 
   for the 'Macro' optimizer.
"""

### --- Project Module Imports --- ###
import physics_engine
import math
from lxml import etree
import re 
from scipy.signal import savgol_filter 
from typing import Tuple, List, Dict, Optional 

# --- GPX Smoothing and Data Cleaning Configuration ---

# _____________________________________________________________________________

# --- ### --- 1. CONFIGURATION --- ### ---

# Smoothing Settings
# GPS elevation data is notoriously noisy. If we don't smooth it, the physics engine will think the rider is constantly riding up and down 20% jagged spikes.
GPX_SMOOTHING_ORDER = 3 # Polynomial order for the filter (3 = Cubic curve fitting).
SMOOTHING_DISTANCE_TARGET_M = 300.0 # We smooth over a window to remove small bumps.
MACRO_SMOOTHING_DISTANCE_TARGET_M = 500.0 # Larger window for detecting general trends (Macro segments).
GRADIENT_CLIP_THRESHOLD = 25.0 # Cap gradients at +/- 25% (ignoring GPS errors that say vertical walls).

# Dynamic Segmentation Settings
# These thresholds define what counts as a "Climb" or "Descent" for strategy planning.
SEGMENT_CLIMB_THRESHOLD = 2.0     # > 2% gradient = "Climb"
SEGMENT_DESCENT_THRESHOLD = -1.5  # < -1.5% gradient = "Descent"
MIN_SEGMENT_LENGTH_M = 250.0      # Don't create a new segment for a 50m bump. Merge it.

# _____________________________________________________________________________

# _____________________________________________________________________________

# --- ### --- 2. MATH & GEOMETRY HELPERS --- ### ---

def calculate_bearing(lat1, lon1, lat2, lon2) -> float:
    """Calculates the compass direction (0-360) from Point A to Point B.
    Essential for calculating Wind Vectors (Headwind vs Crosswind)."""
    
    # 1. Convert Degrees to Radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    # 2. Calculate Longitude Difference
    delta_lon = lon2_rad - lon1_rad
    
    # 3. Calculate Vector Componenets (Spherical Trigonometry)
    # Determines the x (East/West) and y (North/South) componenets
    x = math.sin(delta_lon) * math.cos(lat2_rad) # x represents the East/West component relative to the destination's latitude
    y = math.cos(lat1_rad) * math.sin(lat2_rad) - (math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(delta_lon)) # 'y' represents the North/South component, accounting for the earth's curvature
    
    # 4. Calculate Angle using atan2
    # math.atan2(x, y) computes the angle (azimuth) from the y-axis (North).
    # It handles all four quadrants correctly (NE, SE, SW, NW).
    initial_bearing_rad = math.atan2(x, y)
    
    # 5. Convert back to Degrees
    initial_bearing_deg = math.degrees(initial_bearing_rad)
    
    # 6. Normalize to 0-360 compass degrees
    # atan2 returns values from -180 to +180.
    # The (deg + 360) % 360 trick ensures -90 (West) becomes 270.
    return (initial_bearing_deg + 360) % 360

def calculate_gpx_density(gpx_track_points: List[Dict], total_distance_m: float) -> float:
    """Helper: How many GPS points per kilometer?"""
    if total_distance_m == 0: return 0.0
    
    num_points = len(gpx_track_points)
    total_distance_km = total_distance_m / 1000.0
    
    if total_distance_km == 0: return 0.0
    
    # Simple Density = Count / Distance
    # High Density (e.g. 1000 pts/km) = Needs aggressive smoothing.
    # Low Density (e.g. 10 pts/km) = Needs gentle smoothing.
    return num_points / total_distance_km

def calculate_smoothing_window_size(gpx_density: float, target_smoothing_distance_m: float) -> int:
    """Helper: Calculates required window size to smooth over X meters."""
    if gpx_density <= 0: return 1
    
    # Convert target distance to km to match the density units
    target_smoothing_distance_km = target_smoothing_distance_m / 1000.0
    
    # Math: (Points per km) * (Target km) = Total Points needed
    approx_points = int(gpx_density * target_smoothing_distance_km)
    
    # Constraint 1: Savitzky-Golay filter needs a window size of at least 3.
    window_size = max(3, approx_points)
    
    # Constraint 2: Savitzky-Golay filter requires an ODD window size.
    # If the math gave us an even number (e.g., 50), bump it to 51.
    if window_size % 2 == 0:
        window_size += 1
        
    return window_size

# _____________________________________________________________________________

# _____________________________________________________________________________

# --- ### --- 3. CORE PROCESSING (THE 'ENGINE') --- ### ---
def smooth_elevation_data(gpx_track_points: List[Dict], window_size: int, poly_order: int) -> List[Dict]:
    """
    Applies a Savitzky-Golay filter to smooth elevation data.
    Why Savitzky-Golay? Unlike a simple moving average, it preserves local maxima/minima
    (i.e., it doesn't flatten the top of a hill as much).
    """
    if window_size <= poly_order:
        print(f"Warning: Window size ({window_size}) too small for poly order ({poly_order}). Skipping smoothing.")
        return gpx_track_points
     
    # Extracting the raw data
    elevation_values = [point['ele'] for point in gpx_track_points] # list of the elevation values
    smoothed_elevation = savgol_filter(elevation_values, window_size, poly_order) # applying SG filter
    
    # PASS 1: Assign smoothed elevation to ALL points first. Putting the smoothed values back in the gpx_track_points
    for i, point in enumerate(gpx_track_points):
        point['smoothedElevation'] = smoothed_elevation[i]
        
    # PASS 2: Calculate gradients using the fully populated data (now that we have smoothed data)
    # We use "Rise over Run" (y/x) or (d_ele / d_dist) * 100 
    for i, point in enumerate(gpx_track_points):
        if i < len(gpx_track_points) - 1: # continues until we reach end of gpx_track_points
            delta_elevation = gpx_track_points[i+1]['smoothedElevation'] - gpx_track_points[i]['smoothedElevation'] # Calculate rise (y)
            delta_distance = gpx_track_points[i+1]['cumulativeDistanceM'] - gpx_track_points[i]['cumulativeDistanceM'] # Calculate run (x)
            
            # Calulate gradient
            if delta_distance > 0:
                gradient = (delta_elevation / delta_distance) * 100.0 
            else:
                gradient = 0.0
        else:
            # Last point just copies the previous points gradient (no next point to calc. its gradient)
            gradient = gpx_track_points[i-1]['segmentGradientPercent']
        
        # Save the calculated gradient into the dictionary for the physics engine to use.
        point['segmentGradientPercent'] = gradient
        
    return gpx_track_points

def parse_gpx_file(file_input, force_smoothing_window: Optional[int] = None, auto_downsample: bool = True) -> Dict:
    """
    Reads a GPX file and converts it into a clean list of track points.
    
    Key Steps:
    1. Parse XML.
    2. Handle Namespaces (GPX 1.0 vs 1.1).
    3. Downsample: If file is huge (>3500 pts), thin it out to speed up optimization.
    4. Calculate Distances (Haversine formula).
    5. Smooth Elevations.
    """
    try:
        # 1. INPUT TYPE CHECK
        # This function is flexible: it accepts either a filename (string) OR an actual file object.
        # This allows the script to work locally on your laptop AND on a web server (Streamlit).
        
        if isinstance(file_input, str):
            # CASE A: It's a file path (e.g., "my_ride.gpx"). Open and read it normally.
            with open(file_input, 'r', encoding='utf-8') as f:
                xml_string = f.read()
        else:
            # CASE B: It's a file object (uploaded via web interface).
            # We must reset the "cursor" to the start of the file just in case it was read before.
            if hasattr(file_input, 'seek'):
                file_input.seek(0)
            
            content = file_input.read()
            # If the file was read as raw bytes (computer code), decode it to a text string.
            xml_string = content.decode('utf-8') if isinstance(content, bytes) else content
    
    except Exception as e:
        raise ValueError(f"Could not read GPX input: {e}")
    

    # If the XML string contains an encoding declaration, remove it.
    #xml_string = re.sub(r'<\?xml.*?\?>', '', xml_string)

    # 3. CLEANING XML NAMESPACES (THE "REGEX HACK")
    # GPX files usually start with complex definitions like <gpx xmlns="http://...">.
    # This confuses the Python parser, forcing us to use the full URL every time we look for a tag.
    # This line deletes that definition so we can use simple tag names.
    # xml_string = re.sub(r'\sxmlns="[^"]+"', '', xml_string, count=1)
    xml_string = re.sub(r'<\?xml.*?\?>|\sxmlns="[^"]+"', '', xml_string, count=1)
    
    if not xml_string.strip():
        raise ValueError("GPX file content is empty.")
        
    # 3. PARSE XML STRUCTURE
    # Convert the text string into a structured "Tree" that Python can navigate.
    try:
        root = etree.fromstring(xml_string)
    except etree.XMLSyntaxError as e:
        raise ValueError(f"Invalid XML format: {e}")

    # 4. ROBUST NAMESPACE STRIPPING (THE "ITERATIVE FIX")
    # Even with the regex above, some GPX files attach namespaces to *every single element*.
    # E.g., instead of <trkpt>, they write <{http://www.topografix.com/GPX/1/1}trkpt>.
    # This loop goes through every tag and chops off the {url} prefix.
    # Result: We can reliably find 'trkpt' regardless of GPX version (1.0 or 1.1).
    for elem in root.iter():
        if '}' in elem.tag:
            elem.tag = elem.tag.split('}', 1)[1]
    
    # 5. DATA EXTRACTION
    track_points_raw = []
    
    # Find all Track Points (<trkpt>) anywhere in the file structure
    trkpts = root.findall('.//trkpt')
    if not trkpts:
        raise ValueError("No track points found in GPX file.")
    
    # Loop through points and save the data we care about (Lat, Lon, Elevation)
    for trkpt in trkpts:
        lat = float(trkpt.get('lat'))
        lon = float(trkpt.get('lon'))
        
        # Elevation is a child tag <ele>, not an attribute.
        # Some devices omit elevation; we default to 0.0 to prevent crashes.
        ele_elem = trkpt.find('ele')
        ele = float(ele_elem.text) if ele_elem is not None else 0.0
        track_points_raw.append({'lat': lat, 'lon': lon, 'ele': ele})
        
    # --- DOWNSAMPLING LOGIC ---
    # Optimizing algorithms are slow (require several iterations, increase in data creates exponential increase in time
    # i.e. Reducing 20,000 points to 3,500 makes the Genetic Algorithm 10x faster with minimal accuracy loss.
    was_downsampled = False
    original_count = len(track_points_raw)
    
    MAX_POINTS = 3500 # value gives good balance of accuracy and speed
    total_raw = len(track_points_raw)
    
    if auto_downsample and total_raw > MAX_POINTS:
        step = total_raw // MAX_POINTS # Calculate step to keep roughly MAX_POINTS
        step = max(1, step)  # Ensure step is at least 1 to avoid division by zero errors
        
        print(f"⚠️ Large file detected ({total_raw} pts). Downsampling by factor of {step}.")
 
        # --- THE SLICING MAGIC ---
        # "Take the list 'track_points_raw' from start to finish, but only keep every Nth item (defined by 'step')."
        # A sample downsized from 10,000 to 3,500 would take every third point
        track_points_raw = track_points_raw[::step]
    elif total_raw > MAX_POINTS and not auto_downsample:
        print(f"ℹ️ High Precision Mode: Keeping all {total_raw} points (Performance may suffer).")
   
    # --- DISTANCE CALCULATION (HAVERSINE FORMULA) ---
    # GPS gives us Latitude and Longitude angles. To get meters, we need spherical trigonometry (The Haversine Formula).
    temp_cumulative_distance_m = 0.0
    prev_lat, prev_lon = None, None
    
    for point in track_points_raw:
        if prev_lat is not None:
            R = 6371e3 # Mean Radius of Earth in meters
            
            # Convert degrees to radians (math functions require radians)
            phi1, phi2 = math.radians(prev_lat), math.radians(point['lat'])
            delta_phi = math.radians(lat - prev_lat)
            delta_lambda = math.radians(point['lon'] - prev_lon)
            # Haversine Formula - caluclates distance between two points on a sphere
            a = math.sin(delta_phi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2)**2
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
            
            segment_distance_m = R * c
            temp_cumulative_distance_m += segment_distance_m
        
        # Store the running total on the point object
        point['cumulativeDistanceM'] = temp_cumulative_distance_m
        prev_lat, prev_lon = point['lat'], point['lon']
    
    total_distance_km = temp_cumulative_distance_m / 1000
    num_points = len(track_points_raw)
    
    # --- DYNAMIC WINDOW SIZING ---
    # Adjust smoothing window based on point density.
    # Not all GPS files are the same. Denser GPS data needs a larger window (in points) to cover the same distance (in meters).
    # - File A (1 pt every 10m): Needs a small smoothing window (e.g. 5 points).
    # - File B (1 pt every 1m): Needs a huge window (e.g. 51 points) to cover the same road distance.
    # This logic calculates density to pick the right window size automatically.
    smoothing_window_to_use = 0
    if force_smoothing_window and force_smoothing_window > GPX_SMOOTHING_ORDER:
        smoothing_window_to_use = force_smoothing_window
        print(f"  > Manual override active. Using forced smoothing window of {smoothing_window_to_use}.")
    else:
        # Default minimum window
        dynamic_window = GPX_SMOOTHING_ORDER + 2 
        if total_distance_km > 0.1:
            points_per_km = num_points / total_distance_km
            
            # We want to smooth over approx 'SMOOTHING_DISTANCE_TARGET_M' (e.g. 300m).
            # Math: (Points per km / 1000) * 300
            target_points_in_window = int((points_per_km / 1000) * SMOOTHING_DISTANCE_TARGET_M)
            
            # Filter Requirement: Window size must be ODD.
            if target_points_in_window % 2 == 0:
                target_points_in_window += 1 # Window must be odd for Savitzky-Golay
            
            # Clamp the window: Must be at least 5, and no larger than the file itself.
            dynamic_window = max(GPX_SMOOTHING_ORDER + 2, min(target_points_in_window, num_points - 1))
            
            # Double check oddness after clamping
            if dynamic_window % 2 == 0:
                dynamic_window = dynamic_window -1 if dynamic_window > GPX_SMOOTHING_ORDER + 2 else dynamic_window + 1
        smoothing_window_to_use = dynamic_window
        print(f"  > GPX data density: {points_per_km:.1f} points/km. Using dynamic smoothing window of {smoothing_window_to_use}.")

    # --- APPLY SMOOTHING ---
    # Runs the Savitzky-Golay filter to fix jagged elevation data.
    track_points_smoothed_ele = smooth_elevation_data(track_points_raw, smoothing_window_to_use, GPX_SMOOTHING_ORDER)
    
    # --- FINAL ENRICHMENT LOOP ---
    # Now that we have clean elevation, we rebuild the list one last time to 
    # calculate Gradients and Bearings for every point.
    track_points_final = []
    cumulative_distance_m = 0.0
    prev_lat, prev_lon, prev_ele = None, None, None

    for i, point in enumerate(track_points_smoothed_ele):
        lat, lon, ele = point['lat'], point['lon'], point['ele']
        segment_distance_m = 0.0
        bearing_deg = 0.0
        segment_gradient_percent = 0.0
        
        if prev_lat is not None:
            # Recalculate distance (Haversine again) for the final list
            R = 6371e3  # Earth's radius
            
            # Convert degrees to radians (math functions require radians)
            phi1, phi2 = math.radians(prev_lat), math.radians(lat)
            delta_phi = math.radians(lat - prev_lat)
            delta_lambda = math.radians(lon - prev_lon)
            # Haversine Formula - calculates distance between two points on a sphere
            a = math.sin(delta_phi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2)**2
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
            
            segment_distance_m = R * c
            cumulative_distance_m += segment_distance_m
            
            # Calculate Gradient: (Rise / Run) * 100
            elevation_change_m = ele - prev_ele
            if segment_distance_m > 1e-3: # Avoid divide by zero
                segment_gradient_percent = (elevation_change_m / segment_distance_m) * 100
           
            # --- GRADIENT CLIPPING ---
            # If GPS error says we just rode up a vertical wall (e.g. 50% grade), 
            # cap it at a realistic maximum (25%) so the physics engine doesn't break.
            if segment_gradient_percent > GRADIENT_CLIP_THRESHOLD:
                segment_gradient_percent = GRADIENT_CLIP_THRESHOLD
            elif segment_gradient_percent < -GRADIENT_CLIP_THRESHOLD:
                segment_gradient_percent = -GRADIENT_CLIP_THRESHOLD
            
            # Calculate Bearing (Direction)
            bearing_deg = calculate_bearing(prev_lat, prev_lon, lat, lon)

        track_points_final.append({
            'lat': lat, 'lon': lon, 'ele': ele,
            'smoothedElevation': point.get('smoothedElevation', ele), 
            'cumulativeDistanceM': cumulative_distance_m,
            'segmentDistanceM': segment_distance_m,
            'bearingDeg': bearing_deg,
            'segmentGradientPercent': segment_gradient_percent
        })
        prev_lat, prev_lon, prev_ele = lat, lon, ele

    # Edge Case: The first point has no "previous point" to compare to.
    # We copy the bearing from the second point so the rider doesn't start facing North (0.0) by default.
    if len(track_points_final) > 1:
        track_points_final[0]['bearingDeg'] = calculate_bearing(
            track_points_final[0]['lat'], track_points_final[0]['lon'],
            track_points_final[1]['lat'], track_points_final[1]['lon']
        )
        
    course_name_elem = root.find('.//name')
    course_name = course_name_elem.text if course_name_elem is not None else 'Unnamed Route'
    
    # Prepare data arrays for the plotting libraries (Matplotlib)
    raw_elevations_for_plot = [p['ele'] for p in track_points_raw]
    
    plot_data = {
        'distances_km': [p['cumulativeDistanceM'] / 1000 for p in track_points_final],
        'raw_elevations_m': raw_elevations_for_plot,
        'smoothed_elevations_m': [p['ele'] for p in track_points_final],
        'smoothing_window': smoothing_window_to_use
    }

    return {
        'name': course_name,
        'trackPoints': track_points_final,
        'totalDistanceKm': cumulative_distance_m / 1000,
        'plot_data': plot_data,
        'was_downsampled': was_downsampled,
        'original_point_count': original_count,
        'final_point_count': len(track_points_final)
    }

# _____________________________________________________________________________

# _____________________________________________________________________________

# --- ### --- 4. POST-PROCESSING & ENRICHMENT --- ### ---

def enrich_track_with_surface_profile(gpx_track_points: List[Dict], surface_profile: List[Tuple[float, float]]):
    """
    Applies Rolling Resistance (Crr) values to track points based on distance.
    Allows for courses that change surface (e.g., Road -> Gravel -> Road).
    
    surface_profile format: List of (Distance_km, Crr_value)
    Example: [(0.0, 0.005), (10.0, 0.012)] means "Start at 0.005, switch to 0.012 at 10km"
    """
    # Safety checks
    if not gpx_track_points: return
    
    # If no specific profile is given, assume standard road (Crr 0.005) for the whole ride.
    if not surface_profile:
        for p in gpx_track_points: p['crr'] = 0.005
        return

    # Initialize State Machine
    # Start with the first surface type defined in the list
    current_crr = surface_profile[0][1]
    profile_idx = 0
    next_change_dist_m = float('inf')
    
    # Check if there is a *second* surface type to switch to later
    if len(surface_profile) > 1:
        # Get the distance of the next switch (e.g. 10km) and convert to meters
        next_change_dist_m = surface_profile[1][0] * 1000.0

    # Iterate through every single GPS point
    for p in gpx_track_points:
        dist_m = p['cumulativeDistanceM']
        
        # --- CHECK FOR SURFACE CHANGE ---
        # "Have we ridden past the marker for the next road surface?"
        # We use a 'while' loop here just in case gaps between points are huge 
        # and we skipped multiple surface changes in one step (rare, but possible).
        while dist_m >= next_change_dist_m:
            
            # Advance to the next surface definition
            profile_idx += 1
            
            # If we are still within the list bounds...
            if profile_idx < len(surface_profile):
                current_crr = surface_profile[profile_idx][1]
                
                # Look ahead to the NEXT change
                if profile_idx + 1 < len(surface_profile):
                    next_change_dist_m = surface_profile[profile_idx + 1][0] * 1000.0
                else:
                    # No more changes defined. Stay on this surface forever.
                    next_change_dist_m = float('inf')
            else:
                next_change_dist_m = float('inf')
        
        # Apply the current Rolling Resistance to this specific point
        p['crr'] = current_crr
 
def calculate_total_ascent(track_points: List[Dict]) -> float:
    """Sums all positive elevation changes."""
    total_ascent = 0.0
    
    # Iterate through the list starting at index 1 (so we can look back at i-1)
    for i in range(1, len(track_points)):
        
        # Calculate difference: Current Height - Previous Height
        elevation_change = track_points[i]['ele'] - track_points[i-1]['ele']
        
        # Only care if we went UP.
        # If elevation_change is negative (we went downhill), ignore it.
        if elevation_change > 0:
            total_ascent += elevation_change
            
    return total_ascent
 
# _____________________________________________________________________________

# _____________________________________________________________________________

# --- ### --- 5. ADVANCED SEGMENTATION (THE 'BRAIN') --- ### ---

def get_macro_segment_data(
    gpx_track_points: List[Dict], 
    total_course_distance_m: float,
    rider_params: Optional[Dict] = None, 
    manual_boundaries_km: Optional[List[float]] = None 
) -> Tuple[List[int], List[float], int]:
    
    if not gpx_track_points:
        return [], [], 0

    num_points = len(gpx_track_points)
    
    # Initialize the "Map" array. 
    # If we have 1000 points, this array will eventually look like: 
    # [0, 0, 0, 1, 1, 1, 2, 2, 2...] where numbers are Segment IDs.
    gpx_to_opt_segment_map = [0] * num_points
    macro_segments = []

    # --- OPTION A: MANUAL SEGMENTATION ---
    # Used if the rider says: "I want a split at exactly 10km and 40km."
    # Useful for matching specific race rules or feed zones.
    if manual_boundaries_km and len(manual_boundaries_km) > 0:
        print(f"  > Using MANUAL segmentation boundaries: {manual_boundaries_km}")
        
        boundaries_m = sorted([b * 1000.0 for b in manual_boundaries_km]) # convert to meters
        
        current_boundary_idx = 0
        current_segment_start_idx = 0
        
        # Walk through the points. When we cross a boundary distance, SNIP the segment.
        for i in range(num_points):
            dist = gpx_track_points[i]['cumulativeDistanceM']
            
            # Check if we just crossed a manual split point
            if current_boundary_idx < len(boundaries_m) and dist >= boundaries_m[current_boundary_idx]:
                # Slice the list from Start -> Here
                macro_segments.append(gpx_track_points[current_segment_start_idx:i])
                
                # Reset start pointer for the next segment
                current_segment_start_idx = i
                current_boundary_idx += 1
        
        # Add whatever is left after the final split
        macro_segments.append(gpx_track_points[current_segment_start_idx:])
    
    # --- OPTION B: AUTOMATIC SEGMENTATION (Default) ---
    else:
        # 1. HEAVY SMOOTHING (The "Zoom Out")
        # Standard GPS data is noisy. We apply a MASSIVE smoothing window (500m).
        # # We don't want a 10m flat spot in a 5km climb to be its own segment. we care about the general trend of the hill.
        raw_gradients = [p.get('segmentGradientPercent', 0) for p in gpx_track_points]
        density = calculate_gpx_density(gpx_track_points, total_course_distance_m)
        window = calculate_smoothing_window_size(density, MACRO_SMOOTHING_DISTANCE_TARGET_M) # 500m window
        if window > 3 and len(raw_gradients) > window:
             smoothed_gradients = savgol_filter(raw_gradients, window, 1) 
        else:
             smoothed_gradients = raw_gradients

        # 2. CLASSIFICATION PASS
        # Tag every single point as Uphill (1), Downhill (-1), or Flat (0).
        point_states = []
        for g in smoothed_gradients:
            if g > SEGMENT_CLIMB_THRESHOLD:
                point_states.append(1)  # Climb
            elif g < SEGMENT_DESCENT_THRESHOLD:
                point_states.append(-1)  # Descent
            else:
                point_states.append(0) # Flat
        
        # 3. GROUPING PASS
        # Walk through the list. As long as the "State" (1, -1, 0) stays the same, keep adding points to the current bucket.
        # When the State changes, close the bucket and start a new one.
        temp_segments = []
        if not point_states: return [], [], 0

        current_state = point_states[0]
        current_points = []
        current_len_m = 0.0
        
        for i in range(num_points):
            p = gpx_track_points[i]
            
            # STATE CHANGE DETECTED
            if point_states[i] != current_state:
                # State changed! Save the completed segment.
                temp_segments.append({
                    'state': current_state,
                    'points': current_points,
                    'length_m': current_len_m
                })
                # Reset for the new terrain type
                current_state = point_states[i]
                current_points = []
                current_len_m = 0.0
                
            current_points.append(p)
            current_len_m += p['segmentDistanceM']
            
        # Final segment in the buffer
        temp_segments.append({'state': current_state, 'points': current_points, 'length_m': current_len_m})
        
        # 4. CLEANUP PASS (Merging Tiny Segments - Prevents "Micro-Intervals")
        # Example: A 10km climb has a 50m flat spot. Logic: If a segment is tiny (<250m), just merge it into the previous segment.
        # It's better to treat the whole thing as one big climb than to ask the rider to surge/rest for 10 seconds.
        merged_segments = []
        if temp_segments:
            merged_segments.append(temp_segments[0])
            
            for i in range(1, len(temp_segments)):
                next_seg = temp_segments[i]
                prev_seg = merged_segments[-1]
                
                is_tiny = next_seg['length_m'] < MIN_SEGMENT_LENGTH_M
                
                # MERGE IF:
                # A) The terrain state is actually the same (Flat -> Flat), OR
                # B) The next segment is too small to matter.
                if (prev_seg['state'] == next_seg['state']) or is_tiny:
                    # Merge logic
                    prev_seg['points'].extend(next_seg['points'])
                    prev_seg['length_m'] += next_seg['length_m']
                else:
                    # It's a valid new segment. Keep it.
                    merged_segments.append(next_seg)
        
        # Extract just the raw points for the final output
        macro_segments = [s['points'] for s in merged_segments]
        
    # --- MAP GENERATION ---
    # Create the map: Map[GPX_Point_Index] -> Segment_Index
    # Creates a fast lookup list. If the simulation is at any point, it needs to instantly know: "Am I in segment 1 or 2?"
    
    num_dynamic_segments = len(macro_segments)
    avg_macro_segment_gradients = []
    
    # This counter tracks our absolute position in the original list of ~3500 points
    point_idx = 0
    
    # OUTER LOOP: Go through each "Macro Segment" (e.g., The 5km Climb)
    for seg_idx, segment in enumerate(macro_segments):
        grad_sum = 0.0
        dist_sum = 0.0
        
        # INNER LOOP: Go through every tiny GPS point inside this segment
        for i, p in enumerate(segment):
            
            # Safety Check: Ensure we don't write past the end of the map array
            if point_idx < num_points:
                # THE MAPPING MAGIC:
                # "Point number [point_idx] belongs to Segment [seg_idx]"
                # Example: gpx_to_opt_segment_map[50] = 0 (Point 50 is in Segment 0)
                gpx_to_opt_segment_map[point_idx] = seg_idx
                
                # Move the global cursor forward one step
                point_idx += 1
             
            # --- WEIGHTED AVERAGE MATH ---
            # We can't just average the percentages directly (e.g. (5% + 10%) / 2) 
            # because GPS points are not evenly spaced.
            # A 100m section at 5% matters more than a 1m section at 10%.
            d = p['segmentDistanceM']
            
            # Accumulate "Grade-Meters" (Gradient * Distance)
            grad_sum += p['segmentGradientPercent'] * d
            dist_sum += d
            
        # CALCULATE AVERAGE
        # Formula: Total Grade-Meters / Total Distance
        if dist_sum > 0:
            avg_macro_segment_gradients.append(grad_sum / dist_sum)
        else:
            avg_macro_segment_gradients.append(0.0)
            
    # OUTPUT:
    # 1. The Map (used by Physics Engine to find power targets)
    # 2. The Averages (used by Optimizer to decide strategy)
    # 3. The Count (used to initialize the gene population)
    return gpx_to_opt_segment_map, avg_macro_segment_gradients, num_dynamic_segments

def calculate_macro_segment_metrics(
    simulation_log: List[Dict],
    num_optimization_segments: int,
    optimized_macro_power_profile: List[float],
    final_avg_macro_segment_gradients: List[float],
    total_course_distance_m: float
) -> List[Dict]:
    """
    Aggregates simulation results (second-by-second) into per-segment summaries.
    Used for final reporting (e.g., "Segment 1: Avg Power 300W, Speed 25km/h").
    """
    macro_segment_metrics = []
    
    # 1. CREATE EMPTY BUCKETS
    # We create a list of empty lists. If there are 5 segments, we make 5 buckets.
    # grouped_log_entries = [[], [], [], [], []]
    grouped_log_entries = [[] for _ in range(num_optimization_segments)] # 
    
    # 2. SORT DATA INTO BUCKETS
    # We iterate through every single second of the race log.
    # If a log entry says "I am in Segment 3", we throw it into Bucket #3.
    for entry in simulation_log:
        macro_idx = entry.get('macro_segment_index')
        if macro_idx is not None and 0 <= macro_idx < num_optimization_segments:
            grouped_log_entries[macro_idx].append(entry)

    segment_start_dist_km = 0.0
    
    # 3. ANALYZE EACH BUCKET
    for i in range(num_optimization_segments):
        segment_entries = grouped_log_entries[i]
        
        # --- POWER CALCULATION (REALITY CHECK) ---
        # We prefer to calculate the ACTUAL average power recorded in the log, rather than just displaying the "Target Power".
        # Why? Because of physics overrides. The target might be 300W, but if the rider has to coast for a corner or hits a speed limit, the *actual* average might be 280W.
        if segment_entries:
            # Calculate mean power from the simulation log for this segment
            total_pwr = sum(entry['power'] for entry in segment_entries)
            power = total_pwr / len(segment_entries)
        
        # Fallback: If we have no data (e.g., simulation crashed or didn't reach this segment),
        # just show what the target WAS supposed to be.
        elif i < len(optimized_macro_power_profile):
            # Fallback if no simulation data exists for this segment
            power = optimized_macro_power_profile[i]
        else:
            power = 0.0
            
        # Get the slope (gradient) for this segment so we can display it in the table.
        if i < len(final_avg_macro_segment_gradients):
            avg_gradient = final_avg_macro_segment_gradients[i]
        else:
            avg_gradient = 0.0

        time_taken_s = 0.0
        avg_speed_kmh = 0.0
        actual_segment_distance_km = 0.0
        
        # --- TIME & DISTANCE MATH ---
        if segment_entries:
            # Look at the clock when we entered the segment (First Entry)
            first_entry_time = segment_entries[0]['time']
            # Look at the clock when we left the segment (Last Entry)
            last_entry_time = segment_entries[-1]['time']
            
            # Same for distance markers
            first_entry_distance_km = segment_entries[0]['distance']
            last_entry_distance_km = segment_entries[-1]['distance']
            
            # Delta = End - Start
            time_taken_s = last_entry_time - first_entry_time
            actual_segment_distance_km = last_entry_distance_km - first_entry_distance_km

            # Speed = Distance / Time
            if time_taken_s > 0:
                avg_speed_kmh = physics_engine.calculate_average_speed_kmh(actual_segment_distance_km, time_taken_s)

        # Update the running distance total so the next segment knows where it starts.
        segment_end_dist_km = segment_start_dist_km + actual_segment_distance_km

        # 4. SAVE REPORT CARD
        macro_segment_metrics.append({
            'segment_start_km': segment_start_dist_km,
            'segment_end_km': segment_end_dist_km,
            'avg_gradient': avg_gradient,
            'power': power,
            'time_taken_s': time_taken_s,
            'avg_speed_kmh': avg_speed_kmh,
            'actual_distance_km': actual_segment_distance_km
        })
        
        # The end of this segment is the start of the next one.
        segment_start_dist_km = segment_end_dist_km
        
    return macro_segment_metrics




       
        
        
        
        
        