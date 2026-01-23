#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PHYSICS ENGINE MODULE
---------------------
This is the "Engine Room" of the simulation.
It contains the rules of the road:
1. Physical Constants: Gravity, Air Density, etc.
2. Force Equations: Drag, Friction, Gravity.
3. The Simulator: A loop that calculates the rider's speed second-by-second.
"""

### --- Project Module Imports --- ###
import math
from typing import Tuple, List, Dict, Optional 

# _____________________________________________________________________________

# --- ### --- 1. PHYSICAL CONSTANTS AND SIMULATION CONFIGURATION --- ### ---

# --- Global Constants ---
GRAVITY = 9.81  # Acceleration due to gravity (m/s^2)
RHO_SEA_LEVEL = 1.225 # Air density at sea level (kg/m^3) at 15°C
T_SEA_LEVEL = 288.15 # Standard temperature at sea level (15°C in Kelvin)
TEMP_LAPSE_RATE = 0.0065 # Temperature lapse rate (K/m)
MOLAR_MASS_AIR = 0.0289644 # Molar mass of air (kg/mol)
UNIVERSAL_GAS_CONSTANT = 8.31447 # Universal gas constant (J/(mol·K))

# --- Dynamic Simulation Constants ---
DT = 1.0 # Time Step. We calculate physics once every 1.0 seconds. Lower = more precise, Higher = faster.
REPORT_INTERVAL_DISTANCE_KM = 5.0 # How often the script prints "At 10km..." to the console.
MIN_SPEED_FOR_FORCE_CALC_MPS = 0.5 # Minimum speed (0.5 m/s) to prevent "Divide by Zero" errors when stopped.
BRAKING_DECELERATION_MPS2 = -7.0 # How hard the rider brakes for corners (approx 0.7 G-force). ***

# --- Corner Detection Physics Constants ---
# The model looks ahead to see if a sharp turn is coming.
CORNER_LOOKAHEAD_DISTANCE_M = 30.0 # Rider scans 30 meters down the road for turns.
BEARING_CHANGE_THRESHOLD_DEG = 45.0 # A turn is "sharp" if the road bends more than 45 degrees.
MIN_CORNER_RADIUS_M = 5.0 # The tightest possible turn radius (hairpin).

# --- Dynamic CdA Constants ---
# Adjusts aerodynamics (CdA) based on whether the rider is climbing or descending.
CDA_CLIMBING_THRESHOLD = 4.0  # If gradient > 4%, assume rider sits up (Hands on tops).
CDA_DESCENDING_THRESHOLD = -2.0 # If gradient < -2%, assume rider tucks (Aero position).
CDA_CLIMBING_FACTOR = 1.15    # Climbing penalty: +15% Drag.
CDA_DESCENDING_FACTOR = 0.85  # Descending bonus: -15% Drag.

# --- Gearing Constants (Mechanical Realism) ---
MAX_EFFICIENT_CADENCE_SPEED_KMH = 65.0 # Above this speed, pedaling becomes less effective (spinning out).
ABSOLUTE_MAX_SPEED_KMH = 75.0 # Maximum speed cap (terminal velocity safety).     
 
# _____________________________________________________________________________

# _____________________________________________________________________________

# --- ### --- 2. BASIC UNIT CONVERTERS --- ### ---

def convert_km_to_metres(distance_km: float) -> float:
    """Converts km to meters (Physics standard unit)."""
    return distance_km * 1000

def convert_kmh_to_mps(velocity_kmh: float) -> float:
    """Converts km/h to meters per second (Physics standard unit)."""
    return velocity_kmh / 3.6

def convert_velocity_mps_to_kmh(velocity_mps: float) -> float:
    """Converts meters per second back to km/h (easier to read)."""
    return velocity_mps * 3.6

def convert_gradient_to_radians(gradient_percent: float) -> float:
    """Math Helper: Converts slope percentage (5%) into an angle (radians) for physics formulas."""
    return math.atan(gradient_percent / 100)

def format_time_hms(total_seconds: float) -> Tuple[int, int, int]:
    """Turns raw seconds (e.g., 3665) into a readable time (1h 1m 5s)."""
    if total_seconds == float('inf'):
        return float('inf'), float('inf'), float('inf')
    hours = int(total_seconds // 3600) # Counts the whole hours in the total time
    minutes = int((total_seconds % 3600) // 60) #  After the hours have been counted, it counts the whole minutes left
    seconds = int(total_seconds % 60)  # After the minutes have been counted, it counts the whole seconds left
    return hours, minutes, seconds

# _____________________________________________________________________________

# _____________________________________________________________________________

# --- ### --- 3. PHYSICAL PROPERTIES (AIR & RIDER) --- ### ---

def calculate_total_mass(rider_mass_kg: float, bike_mass_kg: float) -> float:
    """Simple addition: Rider + Bike = Total System Mass."""
    return rider_mass_kg + bike_mass_kg

def calculate_nutrition(total_energy_joules: float, duration_seconds: float, rider_ftp: float) -> Tuple[float, float, str, float]:
    """
    Estimates how much food/carbs the rider needs.
    
    Logic:
    1. Harder riding (High Intensity) = More Carbs burned.
    2. Longer riding = Higher hourly intake recommended.
    
    Returns: Calories burned, Carbs needed, and a text recommendation (e.g., "60-90g/hr").
    """
    if duration_seconds <= 0:
        return 0.0, 0.0, "N/A", 0.0

    # 1. Total Metabolic Cost (1 kJ Mechanical Work ~= 1 kcal Metabolic Cost)
    # Human efficiency is ~20-25%. 1 kcal = 4.184 kJ. 
    # The math cancels out roughly 1:1. (e.g. 1000kJ output / 0.24 eff / 4.184 = 996 kcal)
    total_kcal = total_energy_joules / 1000.0
    
    # 2. Average Power & Intensity
    # Calculate how hard was the ride relative to FTP
    avg_power = total_energy_joules / duration_seconds
    intensity_factor = avg_power / rider_ftp if rider_ftp > 0 else 0.7

    # 3. Estimate Carb vs Fat Burn Ratio (Respiratory Exchange Ratio)
    # Zone 2 (0.6 IF) = ~40% Carbs | Threshold (1.0 IF) = 100% Carbs
    if intensity_factor < 0.6:
        carb_ratio = 0.4
    elif intensity_factor > 1.0:
        carb_ratio = 1.0
    else:
        # Scale linearly between Zone 2 and Threshold
        carb_ratio = 0.4 + ((intensity_factor - 0.6) / 0.4) * 0.6
        
    total_carbs_burned_grams = (total_kcal * carb_ratio) / 4.0 # 4 kcal per gram of carb
    
    # 4. Intake Recommendation (Standard Guidelines)
    # Generating text for carb reccomendation
    duration_hours = duration_seconds / 3600.0
    if duration_hours < 1.25:
        rec_str = "0g (Water/Mouth Rinse)"
        intake_g_per_hr = 0
    elif duration_hours < 2.5:
        rec_str = "30-60g per hour"
        intake_g_per_hr = 45
    elif duration_hours < 4.0:
        rec_str = "60-90g per hour"
        intake_g_per_hr = 75
    else:
        rec_str = "90g+ per hour"
        intake_g_per_hr = 90
        
    total_intake_needed = intake_g_per_hr * duration_hours
    
    return total_kcal, total_carbs_burned_grams, rec_str, total_intake_needed

def calculate_air_density(elevation_m: float, temperature_celsius: float) -> float:
    """
    Calculates how "thick" the air is.
    
    Logic:
    - Higher Altitude = Thinner Air (Less Drag).
    - Higher Temperature = Thinner Air (Less Drag).
    Uses the International Standard Atmosphere (ISA) formula.
    """
    temp_kelvin = temperature_celsius + 273.15 # 273.15 = 0 degrees celsius
    
    # 1. Base density adjusted for temperature
    rho_sea_level_actual_temp = RHO_SEA_LEVEL * (T_SEA_LEVEL / temp_kelvin)
    
    # 2. Adjust for elevation (Air gets thinner as you go up)
    if elevation_m >= 0:
        rho = rho_sea_level_actual_temp * (1 - TEMP_LAPSE_RATE * elevation_m / T_SEA_LEVEL) ** \
              (GRAVITY * MOLAR_MASS_AIR / (UNIVERSAL_GAS_CONSTANT * TEMP_LAPSE_RATE))
    else:
        rho = rho_sea_level_actual_temp
    return rho

def calculate_cda(frontal_area_m2: float, cd_coefficient: float) -> float:
    """Aerodynamic Drag Area (CdA) = Frontal Area * Drag Coefficient."""
    # Currently an unused function as CdA is selected from a drop down menu and then scaled for height
    return frontal_area_m2 * cd_coefficient

def calculate_height_adjusted_cda(base_cda: float, rider_height_cm: float) -> float:
    """
    Adjusts aerodynamics (CdA) for taller or shorter riders.
    Uses 178cm as the standard reference.
    Taller rider = More frontal area = More drag.
    """
    STANDARD_HEIGHT_CM = 178.0
    if rider_height_cm <= 0: return base_cda
    
    scale_factor = (rider_height_cm / STANDARD_HEIGHT_CM) ** 1.5
    return base_cda * scale_factor

def calculate_dynamic_cda(base_cda: float, gradient_percent: float) -> float:
    """
    Adjusts the rider's CdA based on the road slope.
    - Steep Uphill: Rider sits up -> Higher CdA.
    - Steep Downhill: Rider tucks -> Lower CdA.
    """
    if gradient_percent > CDA_CLIMBING_THRESHOLD:
        # Steep uphill
        return base_cda * CDA_CLIMBING_FACTOR 
    elif gradient_percent < CDA_DESCENDING_THRESHOLD:
        # Steep downhill
        return base_cda * CDA_DESCENDING_FACTOR 
    else:
        return base_cda 
 
# _____________________________________________________________________________

# _____________________________________________________________________________

# --- ### --- 4. FORCE GENERATORS (NEWTONIAN PHYSICS) --- ### ---

def calculate_gravity_force(mass_kg: float, gradient_radians: float) -> float:
    """
    Gravity Force:
    - Positive on uphills (pulls you back).
    - Negative on downhills (pushes you forward).
    """
    return mass_kg * GRAVITY * math.sin(gradient_radians)

def calculate_rolling_resistance_force(mass_kg: float, gradient_radians: float, crr: float) -> float:
    """
    Rolling Resistance:
    - Friction from tires on the road.
    - Affected by weight, road angle, and tire quality (Crr).
    - Always slows you down.
    """
    return mass_kg * GRAVITY * math.cos(gradient_radians) * crr

def calculate_drag_force(cda_m2: float, air_density_kg_m3: float, v_relative_to_air_signed: float) -> float:
    """
    Aerodynamic Drag Force:
    - Resistance from moving through air.
    - Increases with the square of speed (going 2x faster = 4x more drag).
    """
    drag_magnitude = 0.5 * cda_m2 * air_density_kg_m3 * (v_relative_to_air_signed ** 2)
    
    # Determines if drag force is against (i.e. headwind/regular air resistance) or with (tailwind) the rider
    if v_relative_to_air_signed > 0:
        return -drag_magnitude 
    elif v_relative_to_air_signed < 0:
        return drag_magnitude  
    else: 
        return 0.0
 
# _____________________________________________________________________________

# _____________________________________________________________________________

# --- ### --- 5. COURSE ANALYSIS (THE 'SENSORS') --- ### ---

def calculate_physics_based_corner_speeds(gpx_track_points, friction_coefficient_mu, rider_skill_factor):
    """
    Scans the GPS track to identify sharp corners.
    
    Logic:
    1. Look ahead 30 meters.
    2. If the bearing changes significantly (>45 degrees), it's a corner.
    3. Calculate the turn radius.
    4. Limit max speed based on friction (tire grip) and physics (centripetal force).
    """
    corner_speed_limits = []
    num_points = len(gpx_track_points)
    if num_points < 2: return []

    # Iterate through every point in the GPX file
    for i in range(num_points - 1):
        p1 = gpx_track_points[i] # The 'Entrance' of the potential corner
        p2_index = i + 1
        dist_so_far = 0.0
        
        # --- LOOKAHEAD LOOP ---
        # We don't compare the next point (1 meter away), because that's too noisy.
        # We look ahead by 'CORNER_LOOKAHEAD_DISTANCE_M' (e.g., 30m) to see the overall road shape.
        while p2_index < num_points:
            # Calculate distance between consecutive points to measure how far we've looked
            dist_seg = gpx_track_points[p2_index]['cumulativeDistanceM'] - gpx_track_points[p2_index-1]['cumulativeDistanceM']
            dist_so_far += dist_seg
            # If we have looked ahead 30m, stop. p2 is now the "Exit" of the corner.
            if dist_so_far >= CORNER_LOOKAHEAD_DISTANCE_M:
                break
            
            # Safety break: Don't look ahead more than 100 points (prevents infinite loops on bad data)
            if p2_index > i + 100: break 
            p2_index += 1
        
        if p2_index >= num_points: break 
         
    
        # --- BEARING ANALYSIS ---
        p2 = gpx_track_points[p2_index] # This is the point 30m down the road
        
        bearing1 = p1['bearingDeg'] # Direction we are facing at the start
        bearing2 = p2['bearingDeg'] # Direction the road is facing 30m later
        
        # Calculate the magnitude of the turn
        delta_bearing = abs(bearing1 - bearing2)
        
        
        # --- COMPASS WRAP-AROUND FIX ---
        # If bearing1 is 350° and bearing2 is 10°, the math says delta is 340°. But in reality, it's only a 20° turn to the right. 
        # This fixes that "crossing North" error.
        if delta_bearing > 180: delta_bearing = 360 - delta_bearing

        # --- GEOMETRY & PHYSICS ---
        # Only process if the turn is sharp enough (defined by constant, usually >45 degrees)
        if delta_bearing >= BEARING_CHANGE_THRESHOLD_DEG:
            
            # The distance along the road acts as the "Chord" of the circle
            chord_distance_m = p2['cumulativeDistanceM'] - p1['cumulativeDistanceM']
            
            if chord_distance_m > 0 and delta_bearing > 0:
                try:
                    # --- RADIUS CALCULATION (Chord Formula) ---
                    # We know the Chord Length (30m) and the Angle (delta_bearing).
                    # Formula: Radius = (Chord / 2) / sin(Angle / 2) 
                    radius_m = (chord_distance_m / 2.0) / math.sin(math.radians(delta_bearing) / 2.0)
                except ValueError:
                    # Protection against math errors (e.g. sin(0))
                    radius_m = MIN_CORNER_RADIUS_M
                
                # Clamp the radius: It physically cannot be tighter than 5 meters (a hairpin)
                radius_m = max(MIN_CORNER_RADIUS_M, radius_m) 
                
                # --- PHYSICS LIMIT (Centripetal Force) ---
                # Formula: v_max = sqrt(mu * g * r)
                # This balances Centripetal Force (mv^2/r) with Tire Friction (mu * mg)
                max_speed_mps = math.sqrt(friction_coefficient_mu * GRAVITY * radius_m)
                
                # --- UPHILL EXCEPTION ---
                # If the corner is uphill (>0.5%), gravity helps us slow down/turn.
                # We triple the speed limit because braking is less critical on climbs.
                if gpx_track_points[i]['segmentGradientPercent'] > 0.5:
                    max_speed_mps *= 3.0 
                
                # Apply the user's skill level (e.g. 0.95 = 95% of theoretical physics limit)
                max_speed_mps *= rider_skill_factor
                max_speed_kmh = convert_velocity_mps_to_kmh(max_speed_mps)
                
                # Mark the "Center" of the corner as the spot where this speed limit applies
                corner_center_distance_m = (p1['cumulativeDistanceM'] + p2['cumulativeDistanceM']) / 2.0
                corner_speed_limits.append((corner_center_distance_m, max_speed_kmh, radius_m, delta_bearing)) # Save it to the list
                
    return corner_speed_limits

def get_course_properties_at_distance(distance_m, gpx_track_points, corner_speeds, total_course_distance_m):
    """
    Finds the gradient and bearing for the rider's current position.
    Also checks if the rider is approaching a known corner.
    """
    # 1. Initialize variables with "Safe Defaults"
    # If something goes wrong or we fall off the map, we assume flat ground (0%), facing North (0 deg), with no speed limit (None).
    current_gradient_percent = 0.0
    current_bearing_deg = 0.0
    current_corner_speed_limit_kmh = None
    current_segment_index = 0 

    # Safety Check: If the map is empty, return the safe defaults immediately.
    if not gpx_track_points: return 0.0, None, 0.0, 0 
    
    # 2. Find "Where am I?" (Linear Search)
    # The GPX file is a list of points (0m, 10m, 20m...). We need to find which two points we are currently between.
    idx = 0
    
    # "Walk" forward through the track points.
    # Keep incrementing 'idx' as long as the NEXT point is still behind us.
    while idx < len(gpx_track_points) - 1 and gpx_track_points[idx+1]['cumulativeDistanceM'] <= distance_m:
        idx += 1
    # When this loop stops, 'gpx_track_points[idx]' is the point we just passed.
    current_segment_index = idx 
    
    # 3. Read Terrain Data
    # Now that we know which index we are at, pull the data from that specific GPX point.
    current_gradient_percent = gpx_track_points[current_segment_index]['segmentGradientPercent']
    current_segment_bearing_deg = gpx_track_points[current_segment_index]['bearingDeg']
    
    # 4. Corner Safety Check
    # Start by assuming we can go infinitely fast (no limit).
    applicable_speed_limit = float('inf')
    # Define the "Danger Zone" size.
    # If the corner is at 500m, and window is 30m, the limit applies from 470m to 530m.
    window_size = CORNER_LOOKAHEAD_DISTANCE_M 
    
    # Check every known corner on the course...
    for corner_center_m, speed_kmh, _, _ in corner_speeds:
        # ...to see if our current distance falls inside its "Danger Zone".
        if (corner_center_m - window_size) <= distance_m < (corner_center_m + window_size):
            # If we are in a danger zone, apply the speed limit.
            # We use 'min' to handle overlapping corners (e.g. S-bends).
            # If Corner A says 30kph and Corner B says 25kph, we must obey the slower one (25kph).
            applicable_speed_limit = min(applicable_speed_limit, speed_kmh)
    
    # If we found a valid limit (i.e., it's no longer infinite), set the variable.
    if applicable_speed_limit != float('inf'):
        current_corner_speed_limit_kmh = applicable_speed_limit

    # 5. Report back to the main physics loop
    return current_gradient_percent, current_corner_speed_limit_kmh, current_segment_bearing_deg, current_segment_index
 
# _____________________________________________________________________________

# _____________________________________________________________________________

# --- ### --- 6. THE MAIN SIMULATION (THE 'BRAIN') --- ### ---

def simulate_course(
    rider_mass: float, bike_mass: float, cda: float, mechanical_efficiency: float,
    starting_elevation_m: float,
    global_wind_speed_kmh: float, global_wind_direction_from_deg: float,
    ambient_temperature_celsius: float,
    enable_cornering_model: bool, friction_coefficient_mu: float, rider_skill_factor: float,
    surface_profile: List[Tuple[float, float]], 
    gpx_track_points: List[Dict], total_course_distance_m: float,
    power_profile_watts: List[float], 
    gpx_to_opt_segment_map: Optional[List[int]] = None, 
    report_progress: bool = True,
    rider_ftp: float = 200.0,
    rider_w_prime: float = 15000.0 
) -> Tuple[float, float, List[Dict], float]: 
    
    """
    THE MAIN SIMULATION LOOP
    ------------------------
    This function moves the rider along the course in 1-second increments.
    
    How it works:
    1. Physics: Calculates forces (Gravity, Drag, Rolling Resistance).
    2. W' Balance: Tracks the rider's anaerobic battery. 
       - If W' hits 0 (Bonk), it rewinds time and forces the rider to go slower.
    3. Integration: Speed += Acceleration * Time.
    """
    
    # 1. Setup Mass & Wind
    total_mass = calculate_total_mass(rider_mass, bike_mass) # Mass that needs to be moved
    GLOBAL_WIND_SPEED_MPS = convert_kmh_to_mps(global_wind_speed_kmh) # Convert wind speed to m/s for calculations

    # 2. DETERMINE STRATEGY MODE
    # The simulator supports two modes:
    # A) "Micro" Mode (Legacy): 1 power value for every single GPS point.
    # B) "Macro" Mode (Optimized): 1 power value per terrain segment (e.g., "Climb 1", "Flat 2").
    
    using_macro_strategy = False
    
    # Check if a segment map was provided (i.e., are we using Macro Mode?)
    if gpx_to_opt_segment_map is not None:
        # Determine how many segments the map expects (e.g., if max index is 14, we need 15 power values)
        required_segments = max(gpx_to_opt_segment_map) + 1
        
        # If the provided power profile matches the number of segments, activate Macro Mode.
        if len(power_profile_watts) == required_segments:
            using_macro_strategy = True

    # 3. PRE-CALCULATE CORNERS
    # Instead of checking for corners every single second (slow), scan the whole course once now.
    corner_speeds_for_sim = [] 
    if enable_cornering_model and gpx_track_points:
        corner_speeds_for_sim = calculate_physics_based_corner_speeds(
            gpx_track_points, friction_coefficient_mu, rider_skill_factor
        )
       
    # 4. INITIALIZE RIDER STATE
    current_distance_m = 0.0
    current_time_seconds = 0.0
    total_energy_joules = 0.0 
    current_velocity_mps = 0.1 # Start with a tiny bit of speed (0.1 m/s).
    # If we start at absolute 0, the force equation F = Power/Velocity creates a "Divide by Zero" error.
    
    # Set starting altitude from the GPX file
    if gpx_track_points:
        starting_elevation_m = gpx_track_points[0]['ele']
    cumulative_elevation_m = starting_elevation_m 

    # 5. INITIALIZE PHYSIOLOGY (THE BATTERY)
    # W' is the "Anaerobic Battery" (energy available above FTP).
    current_w_prime_balance = rider_w_prime 
    min_w_prime_balance = rider_w_prime
    
    # 6. SETUP PROGRESS REPORTING
    # Determine when to print the first status update (e.g., "At 1.0km...").
    if report_progress:
        next_report_distance_m = convert_km_to_metres(REPORT_INTERVAL_DISTANCE_KM)
    else:
        # If reporting is off, set the next report to Infinity so it never triggers.
        next_report_distance_m = float('inf')
        
    # Safety Cap: Stop the simulation if it runs for more than 24 simulated hours.
    MAX_SIMULATION_DURATION_SECONDS = 3600 * 24 
    
    # 7. SETUP THE "REWIND" SYSTEM (CHECKPOINTS)
    # These variables store the state of the simulation at the start of the current segment.
    # If the rider "bonks" (W' < 0), we will revert the simulation to these values.
    checkpoint_dist = 0.0
    checkpoint_time = 0.0
    checkpoint_w_prime = rider_w_prime
    checkpoint_ele = starting_elevation_m
    checkpoint_log_index = 0
    last_strategy_idx = -1
    
    simulation_log = [] 

    # ==========================
    # MAIN TIME LOOP
    # ==========================
    # Keep calculating until we finish the course distance OR hit the 24-hour safety limit.
    while current_distance_m < total_course_distance_m and current_time_seconds < MAX_SIMULATION_DURATION_SECONDS:
        
        # Reset temporary flags for this second
        is_braking = False
        is_coasting = False
        coasting_time_buffer_s = 2.0 # How many seconds we stop pedaling before hitting the brakes

        # 1. Get Terrain Info
        # Ask our sensor function: "What is the gradient, bearing and corner limits right here?"
        gradient_percent, corner_speed_limit_kmh, segment_bearing_deg, current_gpx_segment_index = \
            get_course_properties_at_distance(current_distance_m, gpx_track_points, corner_speeds_for_sim, total_course_distance_m)

        # 2. Get Power Target
        # "Strategy Index" tells us which power target to use.
        # If in Macro mode, this changes only when the terrain changes (e.g., from Flat to Climb).
        if using_macro_strategy:
            current_strategy_idx = gpx_to_opt_segment_map[current_gpx_segment_index]
        else:
            current_strategy_idx = current_gpx_segment_index

        # 3. SAVE GAME (CHECKPOINT)
        # If the strategy index just changed (e.g., we just started a climb), save the state.
        # This allows us to "Rewind" to the bottom of the hill if the rider blows up halfway up.
        if current_strategy_idx != last_strategy_idx:
            checkpoint_dist = current_distance_m
            checkpoint_time = current_time_seconds
            checkpoint_w_prime = current_w_prime_balance
            checkpoint_ele = cumulative_elevation_m
            checkpoint_log_index = len(simulation_log)
            last_strategy_idx = current_strategy_idx

        # 4. SAFETY CHECK: CORNERS
        # We only care about cornering limits if we are going fast enough to crash.
        # If we are climbing a steep hill (> -1%) or riding slowly (< 11.1 m/s), physics handles grip naturally.
        check_for_corners = True
        if gradient_percent > -1.0: check_for_corners = False 
        if current_velocity_mps < 11.1: check_for_corners = False 
        
        if check_for_corners:
            for corner_dist_m, corner_speed_kmh, _, _ in corner_speeds_for_sim:
                # Find the next corner ahead of us
                if corner_dist_m > current_distance_m: 
                    corner_speed_mps = convert_kmh_to_mps(corner_speed_kmh)
                    
                    # If we are currently faster than the safe corner speed:
                    if current_velocity_mps > corner_speed_mps:
                        # Physics: Calculate exactly when we need to use the brakes to slow down in time.
                        # formula: distance = (v^2 - u^2) / 2a
                        required_braking_dist = (corner_speed_mps**2 - current_velocity_mps**2) / (2 * BRAKING_DECELERATION_MPS2)
                        dist_to_corner = corner_dist_m - current_distance_m
                        
                        # Scenario A: PANIC BRAKE. We are at the "Last possible moment".
                        if dist_to_corner <= required_braking_dist:
                            is_braking = True
                            break 
                        # Scenario B: COAST. Stop pedaling a few seconds before braking to be smooth.
                        coasting_dist_threshold = required_braking_dist + (current_velocity_mps * coasting_time_buffer_s)
                        if dist_to_corner <= coasting_dist_threshold:
                            is_coasting = True

        # --- CALCULATE FORCES ---
        
        # Case A: Braking
        if is_braking:
            # Rider is squeezing the brakes hard.
            net_force = total_mass * BRAKING_DECELERATION_MPS2 # F = ma
            simulated_power_watts = 0.0 # Can't pedal while braking
            effective_power_watts = 0.0 
            
            # As the brakes being applied hard dominates all  other forces, the other forces aren't calculated like they are in B and C, this helps saves computational time.
          
        # Case B: Coasting
        elif is_coasting:
            # Rider stops pedaling but doesn't brake yet.
            # Only resistive forces (Drag, Gravity, Friction) apply
            simulated_power_watts = 0.0 
            
            # Recalculate environment for this specific moment
            cumulative_elevation_m = gpx_track_points[current_gpx_segment_index]['ele']
            current_air_density = calculate_air_density(cumulative_elevation_m, ambient_temperature_celsius)
            current_cda = calculate_dynamic_cda(cda, gradient_percent) 
            
            # Wind Math (Vector Analysis)
            relative_wind_angle_rad = math.radians(global_wind_direction_from_deg - segment_bearing_deg)
            headwind_component_mps = GLOBAL_WIND_SPEED_MPS * math.cos(relative_wind_angle_rad)
            crosswind_component_mps = GLOBAL_WIND_SPEED_MPS * math.sin(relative_wind_angle_rad)
            v_relative_to_air_signed = current_velocity_mps + headwind_component_mps
            
            # Calculate Yaw Angle (Crosswind effect)
            # 1. Determine the "Apparent Wind" Angle
            # We compare the sideways wind speed (Crosswind) to thenet forward air speed (Rider Speed + Headwind).
            # - Fast Rider + Light Crosswind = Low Yaw (Wind feels like it's in your face).
            # - Slow Rider + Strong Crosswind = High Yaw (Wind feels like it's from the side).
            # High yaw = more drag (simplified heristic)
            if abs(v_relative_to_air_signed) > 0.1:
                apparent_yaw_rad = math.atan2(abs(crosswind_component_mps), abs(v_relative_to_air_signed))
            else:
                apparent_yaw_rad = 0.0
            # 2. Apply Drag Penalty
            # Most cyclists and bikes are less aerodynamic when air hits them from the side.
            # This formula adds a penalty based on the sine of the angle.
            # - 0 deg Yaw -> Factor 1.0 (No Penalty)
            yaw_factor = 1.0 + 0.5 * (math.sin(apparent_yaw_rad) ** 2)
            current_cda = current_cda * yaw_factor

            # Standard Resistive Forces
            current_crr = gpx_track_points[current_gpx_segment_index].get('crr', 0.005)
            gradient_radians = convert_gradient_to_radians(gradient_percent)
            
            gravity_force = calculate_gravity_force(total_mass, gradient_radians)
            rolling_resistance_force = calculate_rolling_resistance_force(total_mass, gradient_radians, current_crr)
            drag_force = calculate_drag_force(current_cda, current_air_density, v_relative_to_air_signed)
            
            # Net Force = Drag + Gravity + Rolling (No Pedal Power)
            net_force = drag_force - (gravity_force + rolling_resistance_force)
            effective_power_watts = 0.0

        # Case C: Pedaling (normal riding)
        else:
            # 1. Get the target watts from our plan
            target_power_watts = power_profile_watts[min(current_strategy_idx, len(power_profile_watts) - 1)]
            simulated_power_watts = target_power_watts
                
            # 2. Check Mechanical Limits (Spinning Out)
            # You can't put down power if your legs are spinning at 150rpm downhill.
            current_speed_kmh = convert_velocity_mps_to_kmh(current_velocity_mps)
            
            if current_speed_kmh > ABSOLUTE_MAX_SPEED_KMH:
                effective_power_watts = 0.0 
            elif current_speed_kmh > MAX_EFFICIENT_CADENCE_SPEED_KMH:
                # Linearly reduce power efficiency as we approach max speed
                eff_factor = 1.0 - ((current_speed_kmh - MAX_EFFICIENT_CADENCE_SPEED_KMH) / (ABSOLUTE_MAX_SPEED_KMH - MAX_EFFICIENT_CADENCE_SPEED_KMH))
                effective_power_watts = simulated_power_watts * mechanical_efficiency * eff_factor
            else:
                effective_power_watts = simulated_power_watts * mechanical_efficiency
            
            # 3. Recalculate Environment (Same as coasting block above)
            cumulative_elevation_m = gpx_track_points[current_gpx_segment_index]['ele']
            current_air_density = calculate_air_density(cumulative_elevation_m, ambient_temperature_celsius)
            current_cda = calculate_dynamic_cda(cda, gradient_percent) 
            
            relative_wind_angle_rad = math.radians(global_wind_direction_from_deg - segment_bearing_deg)
            headwind_component_mps = GLOBAL_WIND_SPEED_MPS * math.cos(relative_wind_angle_rad)
            crosswind_component_mps = GLOBAL_WIND_SPEED_MPS * math.sin(relative_wind_angle_rad)
            v_relative_to_air_signed = current_velocity_mps + headwind_component_mps
            
            # Calculate Yaw Angle (Crosswind effect)
            # 1. Determine the "Apparent Wind" Angle
            if abs(v_relative_to_air_signed) > 0.1:
                apparent_yaw_rad = math.atan2(abs(crosswind_component_mps), abs(v_relative_to_air_signed))
            else:
                apparent_yaw_rad = 0.0
            # 2. Apply Drag Penalty
            yaw_factor = 1.0 + 0.5 * (math.sin(apparent_yaw_rad) ** 2)
            current_cda = current_cda * yaw_factor
            
            current_crr = gpx_track_points[current_gpx_segment_index].get('crr', 0.005)
            gradient_radians = convert_gradient_to_radians(gradient_percent)
            gravity_force = calculate_gravity_force(total_mass, gradient_radians)
            rolling_resistance_force = calculate_rolling_resistance_force(total_mass, gradient_radians, current_crr)
            drag_force = calculate_drag_force(current_cda, current_air_density, v_relative_to_air_signed)
            
            # 4. Convert Watts to Force
            # P = F * v  =>  F = P / v
            effective_speed_for_force = max(current_velocity_mps, MIN_SPEED_FOR_FORCE_CALC_MPS)
            propulsive_force = effective_power_watts / effective_speed_for_force
            
            # Net Force = Leg Drive - (Drag + Gravity + Rolling Resistance)
            net_force = propulsive_force + drag_force - (gravity_force + rolling_resistance_force)
        
        # --- PHYSIOLOGY UPDATE (THE BATTERY) ---
        # Update W' (Anaerobic Work Capacity) based on Critical Power (CP) model.
        power_delta = simulated_power_watts - rider_ftp
        
        if power_delta > 0:
            # We are sprinting (Above FTP). Draining the battery
            current_w_prime_balance -= (power_delta * DT)
        else:
            # We are recovering (Below FTP). Recharge the battery.
            # Using Skiba (2012) formula for exponential recovery.
            d_cp = rider_ftp - simulated_power_watts
            tau_w_prime = 546.0 * math.exp(-0.01 * d_cp) + 316.0
            current_w_prime_balance = rider_w_prime - (rider_w_prime - current_w_prime_balance) * math.exp(-DT / tau_w_prime)
        
        # Clamp battery (Cannot be > 100%)
        if current_w_prime_balance > rider_w_prime: current_w_prime_balance = rider_w_prime
        if current_w_prime_balance < min_w_prime_balance: min_w_prime_balance = current_w_prime_balance

        # --- THE REWIND MECHANISM (BONK PROTECTION) ---
        # This is the "Safety Valve" for the optimizer.
        # If the optimizer suggests a power that kills the rider (W' < 0), we intervene.
        if current_w_prime_balance < 0:
            
            # We bonked! The optimizer tried to go too hard.
            # Instead of an arbitrary -10% penalty, let's find the PHYSICAL LIMIT.
            
            # 1. Get the power that caused the failure
            w_prime_available = checkpoint_w_prime 
            
            # 2. Adjust Power Downwards
            # How long did we spend in this segment before dying?
            # We can approximate the duration of the segment based on distance and current speed,
            # but a safer heuristic for the Critical Power model is:
            # Max Power = CP + (Available W' / Time)
            # Since we don't know the exact Time yet (circular dependency), we can use a simpler 
            # "Damping" approach that is less punishing than 10%.
            
            # LOGIC: Reduce power to halfway between "Critical Power" and "Failed Power"
            # This gently walks the power down until it barely fits.
            failed_power = power_profile_watts[min(current_strategy_idx, len(power_profile_watts) - 1)]
            
            # If we are above FTP, nudge towards FTP.
            if failed_power > rider_ftp:
                # Find a middle ground between the failure and FTP
                # New Power = average of (Failed Attempt) and (Safe FTP)
                # This converges much faster and stays "high" rather than dropping like a stone.
                new_power = (failed_power + rider_ftp) / 2.0
                
                # If the difference is tiny, just clamp to FTP to stop infinite loops
                if abs(failed_power - new_power) < 1.0:
                    new_power = rider_ftp
                
                power_profile_watts[current_strategy_idx] = new_power
            else:
                # If we bonked BELOW FTP (shouldn't happen with this model, but safety first), just reduce by 5%
                power_profile_watts[current_strategy_idx] *= 0.95

            # 3. Rewind: Reset variables to checkpoint
            current_distance_m = checkpoint_dist
            current_time_seconds = checkpoint_time
            current_w_prime_balance = checkpoint_w_prime
            cumulative_elevation_m = checkpoint_ele
            
            # 4. Delete Log History since Checkpoint
            simulation_log = simulation_log[:checkpoint_log_index]
            
            # 5. Retry Loop
            continue

        # --- MOTION UPDATE ---
        
        # 1. Acceleration = Force / Mass (Newton)
        acceleration_t0 = net_force / total_mass
        
        # 2. Update Distance (d = v * t)
        distance_covered_in_step_m = current_velocity_mps * DT
        current_distance_m += distance_covered_in_step_m
        
        # 3. Update Velocity (v = u + a * t)
        new_velocity_mps = current_velocity_mps + acceleration_t0 * DT
        
        # Hard cap: If physics says we are speeding, but the corner limit says "Slow Down", obey the limit.
        if corner_speed_limit_kmh is not None and check_for_corners:
            corner_limit_mps = convert_kmh_to_mps(corner_speed_limit_kmh)
            if new_velocity_mps > corner_limit_mps: new_velocity_mps = corner_limit_mps
        
        # Sanity check: Can't have negative speed (no reversing).
        if new_velocity_mps < 0: new_velocity_mps = 0.0
        
        # Update global state for next loop
        current_velocity_mps = new_velocity_mps
        current_time_seconds += DT
        total_energy_joules += simulated_power_watts * DT

        # --- LOGGING DATA ---
        simulation_log.append({
            'distance': current_distance_m / 1000,
            'time': current_time_seconds,
            'velocity': convert_velocity_mps_to_kmh(new_velocity_mps),
            'elevation': cumulative_elevation_m,
            'gradient': gradient_percent,
            'power': simulated_power_watts, 
            'macro_segment_index': current_strategy_idx if using_macro_strategy else -1,
            'w_prime': current_w_prime_balance 
        })
        
        # Print progress to console (e.g. "At 5.0km...")
        if report_progress and current_distance_m >= next_report_distance_m:
            vel_kmh = convert_velocity_mps_to_kmh(new_velocity_mps)
            #print(f"  At {current_distance_m/1000:.2f}km: {vel_kmh:.1f}km/h, {simulated_power_watts:.0f}W, W'={current_w_prime_balance/1000:.1f}kJ")
            next_report_distance_m += convert_km_to_metres(REPORT_INTERVAL_DISTANCE_KM)
        
        # Deadlock check: If we are stopped, not pedaling, and on a hill, stop the sim.
        if new_velocity_mps < 0.01 and effective_power_watts <= 0 and gradient_percent >= 0:
            break
        # --- END OF SIMULATION ---
    
    # --- Final Stats ---
    final_total_time_seconds = current_time_seconds
    final_actual_average_power_watts = 0.0
    
    # Only calculate average power if we actually finished the course.
    if current_distance_m >= total_course_distance_m - 0.1 and final_total_time_seconds > 0:
        final_actual_average_power_watts = total_energy_joules / final_total_time_seconds
    else:
        final_total_time_seconds = float('inf') # DNF

    return final_total_time_seconds, final_actual_average_power_watts, simulation_log, min_w_prime_balance
 
# _____________________________________________________________________________

# _____________________________________________________________________________

# --- ### --- 7. POST-RIDE ANALYTICS --- ### ---

def calculate_average_speed_kmh(total_distance_km: float, total_time_seconds: float) -> float:
    """Simple Speed = Distance / Time."""
    if total_time_seconds <= 0: return 0.0
    return total_distance_km / (total_time_seconds / 3600)









