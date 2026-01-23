#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OPTIMIZER MODULE
----------------
This module determines the optimal pacing strategy.
It generates candidate power profiles (e.g., "Ride 300W on the climb, 100W on descent"),
tests them in the Physics Engine, and uses algorithms to find the fastest one.
"""

### --- Project Module Imports --- ###
import physics_engine
import gpx_tool
import random
import time
from typing import Tuple, List, Dict

### --- 1. OPTIMIZER CONFIGURATION,  PARAMETERS  & CONSTANTS--- ###

# --- Terrain Logic ---
"""# These define how the "Heuristic" (Rule-Based) initialization works.
# e.g., "If gradient is > 2.5%, treat it as a steep climb."
SEGMENT_UPHILL_THRESHOLD_PERCENT = 2.5 
SEGMENT_DOWNHILL_THRESHOLD_PERCENT = -1.0 
MIN_SEGMENT_DISTANCE_M = 2000.0"""

# --- Genetic Algorithm (GA) Settings ---
# These values are run in the Web App
# Offline version pulls values from the json file
GA_POPULATION_SIZE = 10 # How many "riders" we simulate per generation. 
GA_NUM_GENERATIONS = 10 # How many times we evolve the population. 
GA_MUTATION_RATE = 0.3 # 30% chance a rider's strategy gets randomly tweaked.
GA_CROSSOVER_RATE = 0.8 # 80% chance two successful riders "breed" to mix strategies.
GA_ELITISM_COUNT = 2 # The top 2 riders survive unchanged to the next generation (prevents regression).
GA_MUTATION_FTP_FACTOR = 0.1 # How much a mutation can change power (e.g. +/- 10% of FTP).

# --- Greedy Algorithm Settings ---
GREEDY_QUICK_ATTEMPTS = 100 
GREEDY_MODERATE_ATTEMPTS = 50000
GREEDY_POWER_SWAP_FTP_FACTOR = 0.01 # Small tweaks (1% FTP) for fine-tuning.
GREEDY_NO_IMPROVEMENT_THRESHOLD = 500 # Stop if we haven't found a better time in 500 tries.  

# --- Constraints (The Rules) ---
# We force the optimizer to be realistic. You can't ride at 500% FTP for an hour.
MAX_POWER_FACTOR = 1.5             # Max allowed power = 120% FTP (for long segments).
MIN_POWER_FACTOR_DOWNHILL = 0.1    # Minimum 10% FTP on descents (don't pedal backwards).
MIN_POWER_FACTOR_NON_DOWNHILL = 0.4 # Minimum 40% FTP on flats (don't stop pedaling).
MAX_POWER_FACTOR_DOWNHILL = 0.7    # Cap downhill effort (aero drag makes high watts wasteful).

# --- Heuristic Thresholds (The "Smart Guess" Rules) --- 
# Defines what counts as a "climb" or "descent" for the initial strategies.
DOWNHILL_GRADIENT_THRESHOLD_FOR_MIN_POWER = -1.5 # Safety: Below -1.5%, enforce min power (prevent pedaling while tucking).
UPHILL_GRADIENT_THRESHOLD_RULE_BASED = 1.5 # Random Init: > 1.5% counts as a climb.
DOWNHILL_GRADIENT_THRESHOLD_RULE_BASED = -1.5 # Random Init: < -1.5% counts as a descent.
FLAT_POWER_FACTOR_RULE_BASED = 0.95 # Random Init: Target 95% FTP on flats.
UPHILL_GRADIENT_THRESHOLD = 1.0 # Strategy Init: > 1.0% triggers specific "Climbing" power targets.
DOWNHILL_GRADIENT_THRESHOLD = DOWNHILL_GRADIENT_THRESHOLD_FOR_MIN_POWER # Strategy Init: Matches the safety threshold.

# --- Heuristic Initialization Factors ---
# Used to create the initial "smart guess" populations.
POWER_FACTOR_UPHILL_AGG = 1.2    # Aggressive: 120% FTP on climbs.
POWER_FACTOR_FLAT_AGG = 1.0      # Aggressive: 100% FTP on flats.
POWER_FACTOR_DOWNHILL_AGG = 0.1  # Aggressive: Rest on descents.

POWER_FACTOR_UPHILL_MOD = 1.05   # Moderate: 105% FTP on climbs.
POWER_FACTOR_FLAT_MOD = 0.95     # Moderate: 95% FTP on flats.
POWER_FACTOR_DOWNHILL_MOD = 0.7  # Moderate: Keep working on descents.

# --- Random Initialization Ranges ---
# When creating random strategies, we pick a value within these ranges.
RANDOM_UPHILL_POWER_FACTOR_MIN = 1.05 # Random Init: Climb between 105%...
RANDOM_UPHILL_POWER_FACTOR_MAX = 1.2 # ...and 120% FTP.

RANDOM_FLAT_POWER_FACTOR_MIN = 0.85 # Random Init: Flat between 85%...
RANDOM_FLAT_POWER_FACTOR_MAX = 1.05 # ...and 105% FTP.

RANDOM_DOWNHILL_POWER_FACTOR_MIN = 0.10 # Random Init: Descent between 10%...
RANDOM_DOWNHILL_POWER_FACTOR_MAX = 0.50 # ...and 50% FTP.

# Values are % of FTP
ZONE_THRESHOLDS = {
    "Zone 7 (Neuromuscular)": 1.51,  
    "Zone 6 (Anaerobic)": 1.21,      
    "Zone 5 (VO2 Max)": 1.06,        
    "Zone 4 (Threshold)": 0.91,        
    "Zone 3 (Tempo)": 0.76,          
    "Zone 2 (Endurance)": 0.56,        
    "Zone 1 (Active Recovery)": 0.00,  
}

# _____________________________________________________________________________

# _____________________________________________________________________________

# --- ### --- 2. DATA TRANSLATION & REPORTING HELPERS --- ### ---

def get_power_zone(power_watts: float, rider_ftp_watts: float) -> str:
    """Helper: categorization of power into standard cycling training zones."""
    
    # Safety Check: Prevent crash if FTP is missing or zero
    if rider_ftp_watts <= 0: 
        return "N/A (FTP not defined)"
    
    # Calculate Intensity Factor (IF) e.g., 300W / 250W FTP = 1.2 (120% Intensity)
    power_factor = power_watts / rider_ftp_watts
    
    # Sort the dictionary from Highest Zone (7) to Lowest Zone (1). 
    # Why? Logic must be "Is it > Zone 7? No. Is it > Zone 6? No..."
    # If we checked Zone 1 first (>0%), every single power value would instantly return "Zone 1".
    sorted_zones = sorted(ZONE_THRESHOLDS.items(), key=lambda item: item[1], reverse=True)
    
    # Iterate down the ladder
    for zone_name, threshold in sorted_zones:
        # If our power matches or exceeds this zone's floor, we belong here.
        if power_factor >= threshold:
            return zone_name
    
    # Fallback (Should typically be covered by Zone 1 at 0.0)
    return "Zone 1 (Active Recovery)" 

def create_full_power_profile_from_macro(optimized_macro_power_profile: List[float], gpx_track_points: List[Dict], gpx_to_opt_segment_map: List[int]) -> List[float]:
    """
    Expands the short list of "Segment Powers" into a long list of "Point-by-Point Powers".
    
    Example:
    - Input (Optimizer): [300W, 100W] (Segment 1 is 300W, Segment 2 is 100W)
    - Map: [0, 0, 0, 1, 1] (Points 0-2 are Seg 1, Points 3-4 are Seg 2)
    - Output (Physics): [300, 300, 300, 100, 100]
    """
    
    full_power_profile = []
    
    # Iterate through every single GPS point in the file (e.g. 3,500 points)
    for gpx_idx in range(len(gpx_track_points)):
        
        # 1. LOOK UP THE SEGMENT
        # Use the pre-calculated map to find out which segment this specific point belongs to.
        macro_segment_idx = gpx_to_opt_segment_map[gpx_idx]
        
        # 2. GET THE POWER TARGET
        # We use 'min' as a safety catch. If the map says "Segment 10" but we only have 
        # 5 power values defined (rare bug), we default to the last known power value to prevent a crash.
        power = optimized_macro_power_profile[min(macro_segment_idx, len(optimized_macro_power_profile) - 1)]
        
        # 3. ASSIGN POWER
        full_power_profile.append(power)
    return full_power_profile

def get_segment_times_from_log(simulation_log: List[Dict], num_segments: int) -> List[float]:
    """
    Parses the simulation results to find out exactly how many seconds 
    were spent in each segment (e.g., Segment 1 took 300s, Segment 2 took 120s).
    """
    
    # Initialize with 0.0 seconds (default)
    segment_times = [0.0] * num_segments
    
    if not simulation_log:
        return segment_times
    
    # 1. CREATE BUCKETS
    # Make an empty list for each segment index.
    grouped_log_entries = [[] for _ in range(num_segments)]
    
    # 2. SORT DATA (The "Sorting Hat")
    # Go through every second of the race log.
    # If the log says "I am in Segment 3", drop that entry into Bucket #3.
    for entry in simulation_log:
        macro_idx = entry.get('macro_segment_index')
        if macro_idx is not None and 0 <= macro_idx < num_segments:
            grouped_log_entries[macro_idx].append(entry)
    
    # 3. CALCULATE DURATION
    for i in range(num_segments):
        segment_entries = grouped_log_entries[i]
        
        # If the rider actually entered this segment...
        if segment_entries:
            # Timestamp of the first entry in the bucket
            start_time = segment_entries[0]['time']
            # Timestamp of the last entry in the bucket
            end_time = segment_entries[-1]['time']
            
            # Duration = Finish - Start
            segment_times[i] = end_time - start_time
            
    return segment_times

# _____________________________________________________________________________

# _____________________________________________________________________________

# --- ### ---  3. CORE CONSTRAINT LOGIC --- ### --- 
def normalize_power_profile_time_weighted(
    power_profile: List[float],
    target_avg_power: float,
    segment_times: List[float],
    max_power_overall: float,
    segment_gradients_avg: List[float],
    rider_ftp_watts: float
) -> List[float]:
    
    """
    Adjusts a raw power profile so that the time-weighted average equals exactly the Target Power.
    This ensures all strategies are 'Fair' (same total energy expenditure).
    """
    
    # 1. SAFETY CHECKS (Input Validation)
    # The optimizer requires three lists of the same length: [Power], [Time], [Gradient]. If the simulation log missed a segment (rare), the lengths might mismatch.
    # These checks fill missing data with safe defaults (1.0 second, 0.0% gradient)to prevent "IndexError: list index out of range" crashes.
    if len(segment_times) < len(power_profile):
        diff = len(power_profile) - len(segment_times)
        segment_times = list(segment_times) + [1.0] * diff
        
    if len(segment_gradients_avg) < len(power_profile):
        diff = len(power_profile) - len(segment_gradients_avg)
        segment_gradients_avg = list(segment_gradients_avg) + [0.0] * diff
    
    # Edge Case: If the profile is empty or the race time is 0 seconds, do nothing.
    if not power_profile or sum(segment_times) < 1e-6:
        return power_profile 

    profile = list(power_profile)
    total_time = sum(segment_times)
    
    # --- PHYSIOLOGICAL LIMITS CALCULATION ---
    # We define dynamic max power limits for each segment based on duration. A rider can hold 500W for 1 minute, but NOT for 20 minutes.
    min_powers = []
    max_powers = []
    
    # Heuristic W' (Anaerobic Capacity) for the optimizer constraints.
    # We use a generous 25,000J here to allow aggressive strategies, 
    # but the physics engine (with the real W') will have the final say.
    SAFE_W_PRIME_J = 25000.0 # "Soft" Anaerobic tank for planning (Engine has "Hard" tank)
    
    for i, g in enumerate(segment_gradients_avg):
        duration_s = max(1.0, segment_times[i])
        
        # 1. Calculate Physiological Ceiling (Critical Power Curve)
        # Formula: Max Sustainable Power = FTP + (W' / Duration)
        # Example: 1 min climb -> FTP + (25000/60) = FTP + 416W (Sprint is OK)
        # Example: 20 min climb -> FTP + (25000/1200) = FTP + 20W (Must be close to Threshold)
        physio_limit_watts = rider_ftp_watts + (SAFE_W_PRIME_J / duration_s)
        
        # 2. Determine Max Power for this segment
        # It cannot exceed the overall max, AND it cannot exceed the physio limit for this duration
        segment_max = min(max_power_overall, physio_limit_watts)
        
        # Set Min/Max bounds based on terrain type
        if g < DOWNHILL_GRADIENT_THRESHOLD_FOR_MIN_POWER:
            # DOWNHILL
            min_powers.append(0.0) # Output = 0W (rider saves energy)
            max_powers.append(rider_ftp_watts * MAX_POWER_FACTOR_DOWNHILL) # Cap for downhill power (bad return on investment, let gravity do the work, save it for the uphill)
        else:
            # UPHILL/FLAT
            min_powers.append(rider_ftp_watts * MIN_POWER_FACTOR_NON_DOWNHILL) # Force a minimum effort so the rider doesn't stop pedaling. Momentum is precious on flats/climbs.
            max_powers.append(segment_max) # Use the physiological limit that's just been calculated. Good place to 'spend watts. We allow the rider to go as hard as their 'Battery' (W') allows

    # --- ITERATIVE BALANCING LOOP ---
    # We push and pull power values until the average matches the target.
    for _ in range(20):  
        # 1. Force values to be within safe bounds
        profile = [max(min_p, min(max_p, p)) for p, min_p, max_p in zip(profile, min_powers, max_powers)]
        
        # 2. Calculate current Energy Expenditure
        current_energy = sum(p * t for p, t in zip(profile, segment_times)) # Total Energy (Work) - sum of power*time
        target_energy = target_avg_power * total_time
        energy_error = target_energy - current_energy
        
        # If we are close enough (<0.5 Joules error), stop.
        if abs(energy_error / total_time) < 0.5: 
            break
        
        # 3. Distribute the Error  
        if energy_error > 0:  
            # --- NEED TO ADD POWER: Prioritize CLIMBS ---
            
            # Find segments that aren't maxed out yet
            adjustable_indices = [i for i, p in enumerate(profile) if p < max_powers[i]]
            if not adjustable_indices: break
            
            # Sort by gradient descending (Steepest to Flattest)
            adjustable_indices.sort(key=lambda i: segment_gradients_avg[i], reverse=True)
            
            remaining_energy_needed = energy_error
            for idx in adjustable_indices:
                if remaining_energy_needed <= 0.01: break
                
                # How much can we add before hitting the ceiling?
                # We are now limited by 'max_powers[idx]', which includes the duration cap
                room_to_grow_watts = max_powers[idx] - profile[idx]
                energy_capacity = room_to_grow_watts * segment_times[idx]
                
                # Add as much as possible
                energy_to_add = min(remaining_energy_needed, energy_capacity)
                power_boost = energy_to_add / segment_times[idx]
                
                profile[idx] += power_boost
                remaining_energy_needed -= energy_to_add

        elif energy_error < 0:  
            # --- NEED TO REDUCE POWER: Remove power from DESCENTS  first (least penalty) ---
            
            # Find segments that aren't at minmum yet
            adjustable_indices = [i for i, p in enumerate(profile) if p > min_powers[i]]
            if not adjustable_indices: break
            
            # Sort by gradient ascending (Steepest Downhill First)
            adjustable_indices.sort(key=lambda i: segment_gradients_avg[i])
            
            energy_to_remove_total = abs(energy_error)
            for idx in adjustable_indices:
                if energy_to_remove_total <= 0.01: break
                
                room_to_cut_watts = profile[idx] - min_powers[idx]
                energy_capacity = room_to_cut_watts * segment_times[idx]
                
                energy_to_remove = min(energy_to_remove_total, energy_capacity)
                power_cut = energy_to_remove / segment_times[idx]
                
                profile[idx] -= power_cut
                energy_to_remove_total -= energy_to_remove

    # Final Clamp to ensure safety after the last adjustment ***
    profile = [max(min_p, min(max_p, p)) for p, min_p, max_p in zip(profile, min_powers, max_powers)]
    
    return profile

# _____________________________________________________________________________

# _____________________________________________________________________________

# --- ### --- 4. GENETIC ALGORITHM OPERATORS --- ### --- 
def initialize_population(num_segments: int, rider_ftp_watts: float, population_size: int, macro_segment_gradients: List[float]) -> List[List[float]]:
    """
    Creates the first generation of strategies ("The Ancestors").
    Instead of being totally random, we seed it with some common sense strategies.
    """
    population = []
    
    # 1. The "Control Group": Constant Power (Flat Pacing)
    population.append([rider_ftp_watts] * num_segments)
    
    # 2. The "Aggressive Climber": Hard on hills, easy on descents
    aggressive_profile = []
    
    # Setting initial power output values for the optimisation based on gradient
    for avg_gradient in macro_segment_gradients:
        if avg_gradient > UPHILL_GRADIENT_THRESHOLD:
            power = rider_ftp_watts * POWER_FACTOR_UPHILL_AGG
        elif avg_gradient < DOWNHILL_GRADIENT_THRESHOLD:
            power = rider_ftp_watts * POWER_FACTOR_DOWNHILL_AGG
        else:
            power = rider_ftp_watts * FLAT_POWER_FACTOR_RULE_BASED 
        
        aggressive_profile.append(power)
    population.append(aggressive_profile)
    
    # 3. The "Moderate Pacer": Same as above but less extreme
    moderate_profile = []
    
    # Setting initial power output values for the optimisation based on gradient
    for avg_gradient in macro_segment_gradients:
        if avg_gradient > UPHILL_GRADIENT_THRESHOLD:
            power = rider_ftp_watts * POWER_FACTOR_UPHILL_MOD
        elif avg_gradient < DOWNHILL_GRADIENT_THRESHOLD:
            power = rider_ftp_watts * POWER_FACTOR_DOWNHILL_MOD
        else:
            power = rider_ftp_watts * FLAT_POWER_FACTOR_RULE_BASED 
        
        moderate_profile.append(power)
    population.append(moderate_profile)
    
    # 4. The "Random Mob": Fill the rest with randomly biased profiles
    while len(population) < population_size:
        biased_random_profile = []
        
        # Setting initial power output values for the optimisation based on gradient
        for avg_gradient in macro_segment_gradients:
            # Pick a random factor within a sensible range for the terrain
            if avg_gradient > UPHILL_GRADIENT_THRESHOLD_RULE_BASED:
                power_factor = random.uniform(RANDOM_UPHILL_POWER_FACTOR_MIN, RANDOM_UPHILL_POWER_FACTOR_MAX)
            elif avg_gradient < DOWNHILL_GRADIENT_THRESHOLD_RULE_BASED: 
                power_factor = random.uniform(RANDOM_DOWNHILL_POWER_FACTOR_MIN, RANDOM_DOWNHILL_POWER_FACTOR_MAX)
            else:
                power_factor = random.uniform(RANDOM_FLAT_POWER_FACTOR_MIN, RANDOM_FLAT_POWER_FACTOR_MAX)
            
            biased_random_profile.append(rider_ftp_watts * power_factor)
        population.append(biased_random_profile)
        
    return population

def mutate(profile: List[float], mutation_rate: float, perturbation_step: float, min_val: float, max_val: float) -> List[float]:
    """
    Randomly modifies a profile. This introduces new traits.
    e.g., "Maybe riding slightly harder on Segment 4 is better?"
    """
    
    # 1. CLONE THE PARENT
    # We create a copy so we don't accidentally modify the original parent profile.
    mutated_profile = list(profile)
    
    # 2. CHECK EVERY SEGMENT (GENE)
    for i in range(len(mutated_profile)):
        
        # 3. ROLL THE DICE
        # mutation_rate (e.g., 0.3) means there is a 30% chance this specific segment gets changed.
        if random.random() < mutation_rate:
            
            # 4. APPLY RANDOM DRIFT
            # Pick a random number between -Step and +Step (e.g., -20W to +20W).
            change = random.uniform(-perturbation_step, perturbation_step)
            
            # 5. APPLY & CLAMP
            # Add the change, but force the result to stay within the safety rules (Min/Max).
            # We don't want a mutation to create negative power or 2000 Watts.
            mutated_profile[i] = max(min_val, min(max_val, mutated_profile[i] + change))
            
    return mutated_profile

def crossover(parent1: List[float], parent2: List[float], crossover_rate: float) -> Tuple[List[float], List[float]]:
    """
    Breeds two strategies together to create offspring.
    Concept: Take the first half of Parent A and the second half of Parent B.
    Hopefully, the child inherits the best parts of both (e.g., A's climbing pace and B's descending pace).
    """
    
    # Safety check: If there's only 1 segment, we can't split it. Return parents as-is.
    # Evolution requires at least 2 genes to mix them up. Return parents as-is.
    if len(parent1) < 2:
        return list(parent1), list(parent2)

    # 1. ROLL THE DICE
    # crossover_rate (e.g. 0.8) means 80% chance these parents will breed, 20% chance they just clone themselves (pass through) to the next generation.
    if random.random() < crossover_rate:
        
        # 2. PICK THE CUT POINT
        # We need to slice the list into two parts. We start at index 1 and end at len-1 to ensure the cut isn't at the very edges.
        # This guarantees the child gets at least one segment from Parent A AND one from Parent B.
        crossover_point = random.randint(1, len(parent1) - 1)
        
        # 3. SPLICE THE GENES
        # Child 1 gets the Start of Parent 1 and the Finish of Parent 2.
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        
        # Child 2 gets the Start of Parent 2 and the Finish of Parent 1.
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        
        return child1, child2
    
    # If the dice roll failed (no breeding), just return copies of the parents.
    return list(parent1), list(parent2)

# _____________________________________________________________________________

# _____________________________________________________________________________

# --- ### --- 5. MAIN OPTIMIZATION ALGORITHMS --- ### --- 

def optimize_pacing_ga(
    rider_params: Dict, sim_params: Dict, course_data: Dict, ga_params: Dict, num_segments: int, progress_bar=None
) -> Tuple[float, List[float], List[int], List[float], List[float], List[Dict]]: 
    
    # 1. UNPACK DATA
    # Extract the raw course and rider data we need for the simulations.
    gpx_track_points = course_data['gpx_track_points']
    total_course_distance_m = course_data['total_course_distance_m']
    rider_target_power_watts = rider_params['rider_target_power_watts']
    rider_ftp = rider_params['rider_ftp_watts']
    
    if not gpx_track_points:
        return float('inf'), [], [], [], [], []

    # 2. GET TERRAIN MAP
    # Ensure we have the "Map" that links GPS points to optimization segments.
    # If it wasn't pre-calculated, calculate it now.
    if 'gpx_to_opt_segment_map' in course_data and 'avg_macro_segment_gradients' in course_data:
        gpx_to_opt_segment_map = course_data['gpx_to_opt_segment_map']
        avg_macro_segment_gradients = course_data['avg_macro_segment_gradients']
    else:
        gpx_to_opt_segment_map, avg_macro_segment_gradients, _ = gpx_tool.get_macro_segment_data(
            gpx_track_points, total_course_distance_m, rider_params=rider_params
        )
        
    # 3. INITIALIZE POPULATION (Generation 0)
    # Create the first batch of riders (e.g., 50 candidates).
    # Some are flat pacers, some are aggressive climbers, some are random.
    max_power_watts_overall = rider_ftp * MAX_POWER_FACTOR
    population = initialize_population(
        num_segments, rider_target_power_watts, ga_params['population_size'], 
        avg_macro_segment_gradients
    )
    
    best_overall_time = float('inf')
    best_overall_profile = []
    final_simulation_log = []

    # Prepare simulation parameters (stripping out GA-specific configs so physics engine doesn't crash)
    sim_rider_params = rider_params.copy()
    sim_rider_params.pop('rider_target_power_watts', None)
    sim_rider_params.pop('rider_ftp_watts', None)
    
    # [CHANGE 3] Extract W' (Anaerobic Capacity) for the physics engine
    rider_w_prime_val = sim_rider_params.pop('w_prime_capacity_j', 14000.0) 
    
    # Define how much mutation changes power (e.g., +/- 25 Watts)
    mutation_step_watts = rider_target_power_watts * ga_params['mutation_ftp_factor']

    generation_history = []
    
    print("--- Starting Pacing Optimisation (Genetic Algorithm with %d generations) ---" % ga_params['num_generations'])
    print(f"  Population Size: {ga_params['population_size']}, Generations: {ga_params['num_generations']}, Mutation Rate (Prob): {ga_params['mutation_rate']*100:.1f}%")
    
    start_time_gen1 = time.time()
    
    # --- EVOLUTION LOOP ---
    # Repeat the cycle of Life, Death, and Rebirth for N generations.
    for generation in range(ga_params['num_generations']):
        
        # Update UI Progress Bar (if running in Streamlit)
        if progress_bar:
            progress = (generation) / ga_params['num_generations']
            progress_bar.progress(progress, text=f"Evolution in progress: Generation {generation + 1}/{ga_params['num_generations']}")
        
        fitness_scores = []
        
        # A. EVALUATE FITNESS (Run the Race)
        for profile in population:
            current_profile = list(profile)
            
            # Start with a dummy assumption that every segment takes 1 second.
            segment_times = [1] * len(current_profile)
            
            # THE "DOUBLE SIM" TRICK
            # We run the simulation twice for every candidate.
            # Pass 1: We guess the segment times. This makes the energy normalization imperfect.
            # Pass 2: We use the *actual* times from Pass 1 to perfectly balance the energy budget.
            for _ in range(2): 
                # Enforce the Energy Budget (Average Power = Target)
                normalized_profile = normalize_power_profile_time_weighted(
                    current_profile, rider_target_power_watts, segment_times, max_power_watts_overall,
                    avg_macro_segment_gradients, rider_target_power_watts
                )
                
                # Clean up params dict for safety
                sim_params_safe = sim_params.copy()
                keys_to_remove = ['gpx_filename', 'ga_population_size', 'ga_num_generations', 'ga_mutation_rate', 'ga_crossover_rate', 'ga_elitism_count', 'ga_mutation_ftp_factor']
                for key in keys_to_remove:
                    sim_params_safe.pop(key, None)
                
                # Run Physics Simulation
                time_taken, avg_power, sim_log, min_w_prime = physics_engine.simulate_course(
                    **sim_rider_params, **sim_params_safe,
                    gpx_track_points=gpx_track_points, total_course_distance_m=total_course_distance_m,
                    power_profile_watts=normalized_profile, # <--- KEY CHANGE: Use normalized_profile directly
                    gpx_to_opt_segment_map=gpx_to_opt_segment_map,
                    report_progress=False,
                    rider_ftp=rider_ftp,
                    rider_w_prime=rider_w_prime_val 
                )
                
                # If simulation failed (rider crashed/bonked badly), abort this candidate
                if time_taken == float('inf'): break
                
                # Update times for the second pass (Precision refinement)
                segment_times = get_segment_times_from_log(sim_log, num_segments)
                current_profile = normalized_profile
                
            # Store the result: (Time, Log, Strategy)
            fitness_scores.append((time_taken, sim_log, current_profile))

            # Track global best
            if time_taken < best_overall_time:
                best_overall_time = time_taken
                best_overall_profile = current_profile
                final_simulation_log = sim_log

        # B. SELECTION (Survival of the Fittest)
        # Sort candidates by Race Time (Lowest/Fastest is first)
        sorted_population_data = sorted(zip(fitness_scores, population), key=lambda pair: pair[0][0])
        sorted_population = [raw_profile for (fitness_tuple, raw_profile) in sorted_population_data]
        
        # Elitism: The fastest few riders survive unchanged to the next generation.
        new_population = [list(p) for p in sorted_population[:ga_params['elitism_count']]]

        # C. BREEDING (Crossover & Mutation)
        # Fill the rest of the population slots with children of the fastest riders.
        while len(new_population) < ga_params['population_size']:
            # Pick two parents from the top 50%
            parent1, parent2 = random.sample(sorted_population[:len(sorted_population)//2], 2)
            
            # Breed them (Mix strategies)
            child1, child2 = crossover(parent1, parent2, ga_params['crossover_rate'])
            
            # Mutate them (Add random variation)
            child1 = mutate(child1, ga_params['mutation_rate'], mutation_step_watts, 0, max_power_watts_overall * 1.2)
            child2 = mutate(child2, ga_params['mutation_rate'], mutation_step_watts, 0, max_power_watts_overall * 1.2)
            
            new_population.append(child1)
            if len(new_population) < ga_params['population_size']:
                new_population.append(child2)
        
        # Replace the old population with the new one
        population = new_population
        
        # D. REPORTING
        generation_history.append(best_overall_time)
        best_h, best_m, best_s = physics_engine.format_time_hms(best_overall_time)
        
        if best_h > 0:
            print(f"  Generation {generation + 1}/{ga_params['num_generations']}: Current Best Time {best_h}h {best_m}m {best_s}s")
        else:
            print(f"  Generation {generation + 1}/{ga_params['num_generations']}: Current Best Time {best_m}m {best_s}s")

        if generation == 0:
            time_gen1 = time.time() - start_time_gen1
            estimated_total_time = time_gen1 * ga_params['num_generations']
            est_h, est_m, est_s = physics_engine.format_time_hms(estimated_total_time)
            if est_h > 0:
                print(f"  Estimated total run time: {est_h}h {est_m}m {est_s}s")
            else:
                print(f"  Estimated total run time: {est_m}m {est_s}s")

    print("--- Optimisation Complete ---")
    final_h, final_m, final_s = physics_engine.format_time_hms(best_overall_time)
    
    # Display hours if applicable
    if final_h > 0:
        print(f"Optimised Time: {final_h}h {final_m}m {final_s}s")
    else:
        print(f"Optimised Time: {final_m}m {final_s}s")
        
    # --- END OF EVOLUTION ---
    
    # Final cleanup and return results
    if not best_overall_profile:
        print("Warning: Genetic Algorithm failed to find a valid optimised profile. Returning a constant power profile as fallback.")
        best_overall_profile = [rider_target_power_watts] * num_segments
    
    # Expand the segment strategy (e.g., 20 values) into the full GPS strategy (e.g., 3500 values)
    optimized_power_profile_full_gpx = create_full_power_profile_from_macro(best_overall_profile, gpx_track_points, gpx_to_opt_segment_map)

    if progress_bar:
        progress_bar.progress(1.0, text="Optimisation Complete!")

    # [FIX 2] Added 'generation_history' to the return statement (7th item)
    return best_overall_time, optimized_power_profile_full_gpx, gpx_to_opt_segment_map, best_overall_profile, avg_macro_segment_gradients, final_simulation_log, generation_history

def optimize_pacing_greedy_hill_climbing(
    rider_params: Dict, sim_params: Dict, course_data: Dict, greedy_params: Dict, num_segments: int, initial_profile_type: str
) -> Tuple[float, List[float], List[int], List[float], List[Dict]]: 
    """
    Optimizes pacing using a 'Greedy' approach.
    1. Start with a baseline strategy.
    2. Randomly move energy from Segment A to Segment B.
    3. If it's faster, keep it. If slower, revert.
    4. Repeat 50,000 times.
    """
    
    # 1. SETUP & UNPACKING
    gpx_track_points = course_data['gpx_track_points'] 
    total_course_distance_m = course_data['total_course_distance_m'] 
    rider_target_power_watts = rider_params['rider_ftp_watts'] # Note: Using FTP as the baseline target
    
    # Greedy configuration
    optimization_attempts = greedy_params['optimization_attempts']  # e.g. 50,000 tries
    power_swap_ftp_factor = greedy_params['power_swap_ftp_factor'] # Size of the swap (e.g. 1% FTP)
    no_improvement_threshold = greedy_params['no_improvement_threshold'] # Stop if stuck for 500 tries

    if not gpx_track_points or num_segments < 2:
        return float('inf'), [], [], [], []
    
    # Ensure we have the map linking GPS points to segments
    gpx_to_opt_segment_map, avg_macro_segment_gradients, _ = gpx_tool.get_macro_segment_data(
        gpx_track_points, total_course_distance_m, rider_params=rider_params
    )
    
    # 2. DEFINE CONSTRAINTS
    # We pre-calculate the Min/Max allowed power for every segment based on terrain.
    # This ensures we don't waste time testing impossible strategies (like sprinting downhill).
    
    best_opt_power_profile = [rider_target_power_watts] * num_segments
    max_power_overall = rider_target_power_watts * MAX_POWER_FACTOR
    
    min_powers = []
    max_powers = []
    for g in avg_macro_segment_gradients:
        if g < DOWNHILL_GRADIENT_THRESHOLD_FOR_MIN_POWER:
             # Downhill: Min 0W (Super Tuck), Max restricted (Aero Drag is high)
            min_powers.append(rider_target_power_watts * MIN_POWER_FACTOR_DOWNHILL)
            max_powers.append(rider_target_power_watts * MAX_POWER_FACTOR_DOWNHILL)
        else:
             # Uphill/Flat: Min 40% FTP (Keep momentum), Max restricted by overall cap
            min_powers.append(rider_target_power_watts * MIN_POWER_FACTOR_NON_DOWNHILL)
            max_powers.append(max_power_overall)

    # Prepare simulation params
    sim_rider_params = rider_params.copy()
    sim_rider_params.pop('rider_ftp_watts', None)
    
    # Extract W' (Anaerobic Capacity) for the physics engine
    rider_w_prime_val = sim_rider_params.pop('w_prime_capacity_j', 14000.0)

    # 3. RUN BASELINE SIMULATION
    # We need a starting point to compare against. Run a flat power profile.
    initial_full_power_profile = create_full_power_profile_from_macro(best_opt_power_profile, gpx_track_points, gpx_to_opt_segment_map)
    best_time, _, initial_sim_log, _ = physics_engine.simulate_course(
        **sim_rider_params, **sim_params,
        gpx_track_points=gpx_track_points, total_course_distance_m=total_course_distance_m,
        power_profile_watts=initial_full_power_profile,
        gpx_to_opt_segment_map=gpx_to_opt_segment_map,
        report_progress=False,
        rider_ftp=rider_target_power_watts,
        rider_w_prime=rider_w_prime_val # [CHANGE 3]
    )
    
    print(f"\n--- Starting Pacing Optimisation (Greedy Time-Weighted Swapping, {optimization_attempts} attempts) ---")
    
    final_simulation_log = initial_sim_log
    # We need segment durations to calculate Energy (Joules = Watts * Seconds)
    current_segment_times = get_segment_times_from_log(initial_sim_log, num_segments)
    no_improvement_count = 0
    power_change_from_target = rider_target_power_watts * power_swap_ftp_factor
    
    # 4. OPTIMIZATION LOOP
    for i in range(optimization_attempts):
        
        # A. SELECTION
        # Pick two random segments: 
        # - Donor: Gives up power
        # - Receiver: Gets extra power
        try:
            donor_idx, receiver_idx = random.sample(range(num_segments), 2)
        except ValueError:
            break

        test_profile = list(best_opt_power_profile)
        
        # Safety check: Valid segment times required for energy math
        if current_segment_times[donor_idx] <= 0 or current_segment_times[receiver_idx] <= 0:
            no_improvement_count += 1
            continue

        # B. ENERGY SWAP MATH
        # Key Concept: We must conserve total energy.
        # If we take 10W from a 10-minute segment (6000 Joules),
        # we can add 100W to a 1-minute segment (6000 Joules).
        energy_change = power_change_from_target * current_segment_times[donor_idx]
        power_increase_for_receiver = energy_change / current_segment_times[receiver_idx]

        # C. BOUNDS CHECK
        # Ensure the swap doesn't violate our Min/Max rules (e.g. negative power).
        if (test_profile[donor_idx] - power_change_from_target >= min_powers[donor_idx] and
            test_profile[receiver_idx] + power_increase_for_receiver <= max_powers[receiver_idx]):
            
            # Apply the swap
            test_profile[donor_idx] -= power_change_from_target
            test_profile[receiver_idx] += power_increase_for_receiver
            
            # D. TEST THE NEW STRATEGY
            full_test_profile = create_full_power_profile_from_macro(test_profile, gpx_track_points, gpx_to_opt_segment_map)
            new_time, avg_power, new_sim_log, _ = physics_engine.simulate_course( 
                **sim_rider_params, **sim_params,
                gpx_track_points=gpx_track_points, total_course_distance_m=total_course_distance_m,
                power_profile_watts=full_test_profile,
                gpx_to_opt_segment_map=gpx_to_opt_segment_map,
                report_progress=False,
                rider_ftp=rider_target_power_watts,
                rider_w_prime=rider_w_prime_val # [CHANGE 3]
            )
            
            # E. EVALUATE ("Hill Climbing")
            if new_time < best_time:
                # SUCCESS: The new strategy is faster. Keep it.
                best_time = new_time
                best_opt_power_profile = test_profile
                final_simulation_log = new_sim_log
                # Update times because riding faster changes segment durations
                current_segment_times = get_segment_times_from_log(new_sim_log, num_segments)
                no_improvement_count = 0
            else:
                # FAILURE: The new strategy is slower. Discard it.
                no_improvement_count += 1
        else:
           # Swap was illegal (violated bounds). Skip.
           no_improvement_count += 1

        # Progress logging
        if (i + 1) % 10 == 0:
            best_min, best_sec = physics_engine.format_time_hms(best_time)[1:]
            print(f"  Attempt {i + 1}/{optimization_attempts}: Current Best Time {best_min}m {best_sec}s")
        
        # F. STOPPING CONDITION
        # If we try 500 times and can't find a better move, assume we are at the peak.
        if no_improvement_count >= no_improvement_threshold:
            print(f"  No improvement in {no_improvement_threshold} attempts. Ending early.")
            break
            
    print("--- Optimisation Complete ---")
    final_optimized_time_hms = physics_engine.format_time_hms(best_time)
    print(f"Optimised Time: {final_optimized_time_hms[1]}m {final_optimized_time_hms[2]}s")
    optimized_power_profile_full_gpx = create_full_power_profile_from_macro(best_opt_power_profile, gpx_track_points, gpx_to_opt_segment_map)
    
    return best_time, optimized_power_profile_full_gpx, gpx_to_opt_segment_map, best_opt_power_profile, final_simulation_log

# _____________________________________________________________________________










