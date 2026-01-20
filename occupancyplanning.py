import cv2
import numpy as np
import heapq
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, TypedDict

class DetectedObject(TypedDict):
    label: str
    role: str
    priority: int
    metadata: Optional[dict]

class MaskMetadata(TypedDict):
    bbox: List[int]
    area: float
    predicted_iou: float
    stability_score: float

class SAMMask(TypedDict):
    mask: np.ndarray
    metadata: MaskMetadata

# --- IMPROVED A* WITH TIMEOUT ---
def heuristic(a, b):
    return np.sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2)

def find_nearest_free(grid: np.ndarray, point: Tuple[int, int], max_dist: int = 50) -> Tuple[int, int]:
    """Find nearest non-obstacle pixel with REDUCED search radius"""
    r, c = int(point[0]), int(point[1])
    r = max(0, min(r, grid.shape[0] - 1))
    c = max(0, min(c, grid.shape[1] - 1))
    
    if grid[r, c] == 0:
        return (r, c)
    
    # Reduced search radius for speed
    for d in range(1, min(max_dist, 50)):
        for dr in range(-d, d + 1):
            for dc in [-d, d]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                    if grid[nr, nc] == 0:
                        return (nr, nc)
    
    print(f"  [WARNING] Could not find free space near {point}")
    return (r, c)

def astar(occupancy_grid: np.ndarray, cost_grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int], max_iterations: int = 50000) -> Optional[List[Tuple[int, int]]]:
    """
    A* with MAX_ITERATIONS limit to prevent infinite loops
    """
    # Nudge to free space
    start = find_nearest_free(occupancy_grid, start, max_dist=30)
    goal = find_nearest_free(occupancy_grid, goal, max_dist=30)
    
    if occupancy_grid[start] == 1:
        print(f"  [ERROR] Start {start} is blocked, using emergency path")
        return get_line_points(start, goal)
    
    if occupancy_grid[goal] == 1:
        print(f"  [ERROR] Goal {goal} is blocked, using emergency path")
        return get_line_points(start, goal)
    
    neighbors = [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]
    close_set = set()
    came_from = {}
    gscore = {start: 0}
    fscore = {start: heuristic(start, goal)}
    oheap = []
    
    heapq.heappush(oheap, (fscore[start], start))
    
    iterations = 0
    while oheap and iterations < max_iterations:
        iterations += 1
        current = heapq.heappop(oheap)[1]
        
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            print(f"  [SUCCESS] Path found in {iterations} iterations")
            return path[::-1]
        
        close_set.add(current)
        
        for i, j in neighbors:
            neighbor = (current[0] + i, current[1] + j)
            
            # Bounds check
            if not (0 <= neighbor[0] < occupancy_grid.shape[0] and 0 <= neighbor[1] < occupancy_grid.shape[1]):
                continue
            
            # Obstacle check
            if occupancy_grid[neighbor[0], neighbor[1]] == 1:
                continue
            
            dist_cost = 1.414 if (i != 0 and j != 0) else 1.0
            total_step_cost = dist_cost * cost_grid[neighbor[0], neighbor[1]]
            
            tentative_g_score = gscore[current] + total_step_cost
            
            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, float('inf')):
                continue
            
            if tentative_g_score < gscore.get(neighbor, float('inf')):
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(oheap, (fscore[neighbor], neighbor))
    
    # Timeout or no path found
    print(f"  [TIMEOUT] A* stopped after {iterations} iterations - using direct path")
    return get_line_points(start, goal)

def get_line_points(p1, p2):
    """Bresenham line for emergency fallback"""
    r0, c0 = int(p1[0]), int(p1[1])
    r1, c1 = int(p2[0]), int(p2[1])
    num_points = int(max(abs(r1 - r0), abs(c1 - c0)))
    if num_points == 0:
        return [p1]
    
    rs = np.linspace(r0, r1, num_points + 1)
    cs = np.linspace(c0, c1, num_points + 1)
    return list(zip(rs, cs))

class DisasterPlanner:
    def __init__(self, occupancy_grid: np.ndarray, cost_grid: np.ndarray):
        self.occupancy_grid = occupancy_grid
        self.cost_grid = cost_grid

    @classmethod
    def build_from_perception(cls, grid_shape: Tuple[int, int], detections: List[DetectedObject], masks: List[SAMMask]):
        """
        IMPROVED FUSION: Actually use VLM semantics to label SAM masks
        """
        occupancy = np.zeros(grid_shape, dtype=np.uint8)
        cost = np.ones(grid_shape, dtype=np.float32)
        
        print(f"\n=== PERCEPTION FUSION ===")
        print(f"VLM detections: {len(detections)}")
        print(f"SAM masks: {len(masks)}")
        
        # Step 1: Categorize VLM detections by role
        obstacles = [d for d in detections if d['role'] == 'obstacle']
        hazards = [d for d in detections if d['role'] == 'hazard']
        clear_areas = [d for d in detections if d['role'] == 'clear']
        
        print(f"  Obstacles: {len(obstacles)}")
        print(f"  Hazards: {len(hazards)}")
        print(f"  Clear areas: {len(clear_areas)}")
        
        # Step 2: Apply semantic rules to SAM masks
        grid_area = grid_shape[0] * grid_shape[1]
        
        for idx, m_obj in enumerate(masks):
            mask = m_obj["mask"]
            area = m_obj["metadata"]["area"]
            area_ratio = area / grid_area
            
            # RULE 1: Very large masks (>5% of image) = likely ROADS or OPEN TERRAIN
            if area_ratio > 0.05:
                if clear_areas:  # VLM says roads exist
                    # Mark as LOW COST (easy to traverse)
                    cost[mask > 0] = 0.1
                    print(f"  Mask {idx}: ROAD/CLEAR (area {area_ratio:.1%})")
                else:
                    # Large but not identified as clear = background
                    cost[mask > 0] = 1.0
            
            # RULE 2: Medium masks (0.1% - 5%) = likely BUILDINGS
            elif 0.001 < area_ratio <= 0.05:
                if obstacles or not clear_areas:
                    # Mark as OBSTACLE
                    occupancy[mask > 0] = 1
                    cost[mask > 0] = 100.0
                    print(f"  Mask {idx}: OBSTACLE (area {area_ratio:.1%})")
            
            # RULE 3: Small masks (< 0.1%) = details, ignore or low penalty
            else:
                cost[mask > 0] = 2.0
        
        # Step 3: If VLM detected hazards, increase cost globally
        if hazards:
            print(f"  Applying hazard penalty across map")
            cost = cost * 1.5
        
        # Smooth cost for natural paths
        cost = cv2.GaussianBlur(cost, (5, 5), 0)
        
        print(f"Occupancy coverage: {(occupancy > 0).sum() / grid_area * 100:.1f}%")
        print(f"=== FUSION COMPLETE ===\n")
        
        return cls(occupancy, cost)

    def save_occupancy_grid(self, save_path: str):
        """Save binary occupancy map"""
        grid_img = (self.occupancy_grid * 255).astype(np.uint8)
        cv2.imwrite(save_path, grid_img)
        print(f"Occupancy grid saved: {save_path}")

    def plan_mission(self, start: Tuple[int, int], targets: List[Tuple[int, int]], base: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        GREEDY TSP with timeout protection
        """
        current_pos = start
        remaining_targets = list(targets)
        full_path = []
        
        print(f"\n=== MISSION PLANNING ===")
        print(f"Start: {start}")
        print(f"Targets: {len(targets)}")
        print(f"Base: {base}")
        
        segment_count = 0
        max_segments = 20  # Safety limit
        
        while remaining_targets and segment_count < max_segments:
            segment_count += 1
            
            # Find nearest target
            nearest_idx = 0
            min_dist = float('inf')
            for i, target in enumerate(remaining_targets):
                dist = heuristic(current_pos, target)
                if dist < min_dist:
                    min_dist = dist
                    nearest_idx = i
            
            next_target = remaining_targets.pop(nearest_idx)
            print(f"  Segment {segment_count}: {current_pos} -> {next_target}")
            
            segment = astar(self.occupancy_grid, self.cost_grid, current_pos, next_target, max_iterations=30000)
            
            if segment:
                full_path.extend(segment)
                current_pos = next_target
            else:
                print(f"  [WARNING] Failed to reach target {next_target}")
        
        # Return to base
        print(f"  Final leg: {current_pos} -> {base}")
        final_leg = astar(self.occupancy_grid, self.cost_grid, current_pos, base, max_iterations=30000)
        if final_leg:
            full_path.extend(final_leg)
        
        print(f"=== PATH COMPLETE: {len(full_path)} waypoints ===\n")
        return full_path

    def smooth_path(self, path: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Simplify path with line-of-sight checks"""
        if len(path) < 3:
            return path
        
        smoothed = [path[0]]
        current_idx = 0
        
        while current_idx < len(path) - 1:
            furthest_visible = current_idx + 1
            
            for look_ahead in range(current_idx + 2, min(current_idx + 50, len(path))):
                p1 = path[current_idx]
                p2 = path[look_ahead]
                
                if self.has_line_of_sight(p1, p2):
                    furthest_visible = look_ahead
                else:
                    break
            
            smoothed.append(path[furthest_visible])
            current_idx = furthest_visible
        
        return smoothed

    def has_line_of_sight(self, p1, p2) -> bool:
        """Check if straight line is obstacle-free"""
        points = get_line_points(p1, p2)
        for r, c in points:
            r, c = int(r), int(c)
            if 0 <= r < self.occupancy_grid.shape[0] and 0 <= c < self.occupancy_grid.shape[1]:
                if self.occupancy_grid[r, c] == 1:
                    return False
        return True

    def visualize_results(self, perception_img: np.ndarray, full_path: List[Tuple[int, int]], 
                         targets: List[Tuple[int, int]], save_path: str = "result.png", 
                         start_point: Tuple[int, int] = (50, 50), end_point: Tuple[int, int] = None):
        """Render final path visualization"""
        print(f"Rendering path visualization...")
        
        path_to_draw = self.smooth_path(full_path) if len(full_path) > 2 else full_path
        
        out = perception_img.copy()
        out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(15, 10))
        
        if path_to_draw and len(path_to_draw) > 1:
            pts = np.array([[p[1], p[0]] for p in path_to_draw], np.int32)
            cv2.polylines(out, [pts], False, (0, 255, 0), 10)
            cv2.polylines(out, [pts], False, (0, 0, 0), 2)
        
        plt.imshow(out)
        
        # Plot targets
        for t in targets:
            plt.scatter(t[1], t[0], color='red', s=400, marker='x', linewidths=3)
        
        # Start point
        plt.scatter(start_point[1], start_point[0], color='blue', s=500, marker='o', 
                   edgecolors='white', linewidths=2)
        
        # End point
        ep = end_point if end_point else start_point
        plt.scatter(ep[1], ep[0], color='green', s=600, marker='D', 
                   edgecolors='white', linewidths=2)
        
        plt.title("Drone Rescue Mission Path")
        plt.axis('off')
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"Visualization saved: {save_path}")
