import cv2
import numpy as np
import heapq
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, TypedDict

# Inlined Types for compatibility
class DetectedObject(TypedDict):
    label: str
    role: str # 'target', 'obstacle', 'hazard'
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

# --- RISK-AWARE A* ALGORITHM ---
def heuristic(a, b):
    return np.sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2)

def find_nearest_free(grid: np.ndarray, point: Tuple[int, int], max_dist: int = 100) -> Tuple[int, int]:
    """Radial search for the nearest non-obstacle pixel."""
    r, c = int(point[0]), int(point[1])
    # Ensure starting point is clipped to bounds
    r = max(0, min(r, grid.shape[0] - 1))
    c = max(0, min(c, grid.shape[1] - 1))
    
    if grid[r, c] == 0:
        return (r, c)
        
    # Search in expanding diamonds/squares
    for d in range(1, max_dist):
        for dr in range(-d, d + 1):
            # Vertical edges
            for dc in [-d, d]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                    if grid[nr, nc] == 0: return (nr, nc)
            # Horizontal edges (excluding corners already covered by vertical edges)
            for dc in range(-d + 1, d):
                nr, nc = r + dr, c + dc
                if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                    if grid[nr, nc] == 0: return (nr, nc)
                    
    print(f"  [Planner] Failed to find free space near {point} after {max_dist}px search.")
    return (r, c)

def astar(occupancy_grid: np.ndarray, cost_grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
    # 1. Bounds and Nudging
    if not (0 <= start[0] < occupancy_grid.shape[0] and 0 <= start[1] < occupancy_grid.shape[1]):
        print(f"  [Planner Error]: Start {start} out of bounds.")
        return None
    
    # Nudge if blocked
    start = find_nearest_free(occupancy_grid, start)
    goal = find_nearest_free(occupancy_grid, goal)
        
    if occupancy_grid[start] == 1:
        print(f"  [Planner Warning]: Start {start} still blocked after recovery. Forcing line.")
        return DisasterPlanner.get_line_points(start, goal)
    if occupancy_grid[goal] == 1:
        print(f"  [Planner Warning]: Goal {goal} blocked. Forcing line.")
        return DisasterPlanner.get_line_points(start, goal)
        
    neighbors = [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]
    close_set = set()
    came_from = {}
    gscore = {start: 0}
    fscore = {start: heuristic(start, goal)}
    oheap = []
    
    heapq.heappush(oheap, (fscore[start], start))
    
    while oheap:
        current = heapq.heappop(oheap)[1]
        
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            return path[::-1]
            
        close_set.add(current)
        for i, j in neighbors:
            neighbor = (current[0] + i, current[1] + j)
            
            if 0 <= neighbor[0] < occupancy_grid.shape[0] and 0 <= neighbor[1] < occupancy_grid.shape[1]:
                if occupancy_grid[neighbor[0], neighbor[1]] == 1: 
                    continue
            else: 
                continue
                
            dist_cost = 1.414 if i != 0 and j != 0 else 1.0
            total_step_cost = dist_cost * cost_grid[neighbor[0], neighbor[1]]
            
            tentative_g_score = gscore[current] + total_step_cost
            
            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0): 
                continue
                
            if tentative_g_score < gscore.get(neighbor, float('inf')):
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(oheap, (fscore[neighbor], neighbor))
                
    # Fallback to straight line if A* fails
    print("  [CRITICAL]: EMERGENCY FALLBACK - Straight line path forced through obstacles.")
    return DisasterPlanner.get_line_points(start, goal)

class DisasterPlanner:
    def __init__(self, occupancy_grid: np.ndarray, cost_grid: np.ndarray):
        self.occupancy_grid = occupancy_grid
        self.cost_grid = cost_grid

    @classmethod
    def build_from_perception(cls, grid_shape: Tuple[int, int], detections: List[DetectedObject], masks: List[SAMMask]):
        """
        Streamlined fusion: Builds occupancy and cost grids from VLM and SAM.
        """
        occupancy = np.zeros(grid_shape, dtype=np.uint8)
        cost = np.ones(grid_shape, dtype=np.float32)
        grid_area = grid_shape[0] * grid_shape[1]
        
        print(f"Fusing perception: {len(detections)} objects, {len(masks)} masks...")
        
        # Simple Logic: Pair VLM roles with SAM geometry
        for m_obj in masks:
            mask = m_obj["mask"]
            area = m_obj["metadata"]["area"]
            
            # Heuristic assignment
            # 1. OBSTACLES (Buildings, Houses, Trees): 
            # Small to medium segments are usually structures
            if (grid_area * 0.0001 < area < grid_area * 0.04):
                occupancy = np.maximum(occupancy, mask.astype(np.uint8))
                cost += mask.astype(np.float32) * 50.0 # High cost for obstacles
                
            # 2. CLEAR TERRAIN (Roads, Open Fields):
            # Very large segments are likely roads or background
            elif (area >= grid_area * 0.04):
                # Explicitly keep it out of occupancy and LOWER cost
                cost -= mask.astype(np.float32) * 5.0
                
        # Smooth cost for natural paths
        cost = cv2.GaussianBlur(cost, (5, 5), 0)
        return cls(occupancy, cost)

    def save_occupancy_grid(self, save_path: str):
        """Saves a binary image (White = Obstacle, Black = Safe)"""
        # Map: 1 (obstacle) -> 255 (white), 0 (free) -> 0 (black)
        grid_img = (self.occupancy_grid * 255).astype(np.uint8)
        cv2.imwrite(save_path, grid_img)
        print(f"Occupancy grid saved to {save_path}")

    def plan_mission(self, start: Tuple[int, int], targets: List[Tuple[int, int]], base: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Solves multi-target rescue path using a simple greedy TSP approach.
        """
        current_pos = start
        remaining_targets = list(targets)
        full_path = []
        
        print(f"Planning rescue path for {len(targets)} targets...")
        
        from tqdm import tqdm
        pbar = tqdm(total=len(targets) + 1, desc="Path Planning")
        
        while remaining_targets:
            # Find nearest target
            nearest_idx = 0
            min_dist = float('inf')
            for i, target in enumerate(remaining_targets):
                dist = heuristic(current_pos, target)
                if dist < min_dist:
                    min_dist = dist
                    nearest_idx = i
            
            next_target = remaining_targets.pop(nearest_idx)
            print(f"  [Planner]: Planning segment to target at {next_target}...")
            segment = astar(self.occupancy_grid, self.cost_grid, current_pos, next_target)
            
            if segment:
                full_path.extend(segment)
                current_pos = next_target
                print(f"  [Planner]: Success. Segment added.")
            else:
                print(f"  [Planner Warning]: Failed to reach target at {next_target}.")
            
            pbar.update(1)

        # Return to base
        final_leg = astar(self.occupancy_grid, self.cost_grid, current_pos, base)
        if final_leg:
            full_path.extend(final_leg)
        
        pbar.update(1)
        pbar.close()
            
        return full_path

    def smooth_path(self, path: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Simplifies the path by checking for direct line-of-sight between points.
        This produces a "straight where possible, curved where needed" effect.
        """
        if len(path) < 3:
            return path
            
        smoothed = [path[0]]
        current_idx = 0
        
        while current_idx < len(path) - 1:
            # Look ahead for the furthest point we can see directly
            furthest_visible = current_idx + 1
            for look_ahead in range(current_idx + 2, len(path)):
                p1 = path[current_idx]
                p2 = path[look_ahead]
                
                # Simple line-of-sight check
                if self.has_line_of_sight(p1, p2):
                    furthest_visible = look_ahead
                else:
                    # Once we hit an obstacle, the previous point was the furthest visible
                    break
            
            smoothed.append(path[furthest_visible])
            current_idx = furthest_visible
            
        return smoothed

    def has_line_of_sight(self, p1, p2) -> bool:
        """Check if a straight line between p1 and p2 hits any obstacles."""
        points = self.get_line_points(p1, p2)
        for r, c in points:
            if self.occupancy_grid[int(r), int(c)] == 1:
                return False
        return True

    @staticmethod
    def get_line_points(p1, p2):
        """Bresenham-like line point generation."""
        r0, c0 = p1
        r1, c1 = p2
        num_points = int(max(abs(r1 - r0), abs(c1 - c0)))
        if num_points == 0: return [p1]
        
        rs = np.linspace(r0, r1, num_points + 1)
        cs = np.linspace(c0, c1, num_points + 1)
        return list(zip(rs, cs))

    def visualize_results(self, perception_img: np.ndarray, full_path: List[Tuple[int, int]], targets: List[Tuple[int, int]], save_path: str = "result.png", start_point: Tuple[int, int] = (50, 50), end_point: Tuple[int, int] = None):
        print(f"Rendering refined path and saving to {save_path}...")
        
        # Smooth the path for a cleaner "straight and curved" look
        path_to_draw = self.smooth_path(full_path)
        
        out = perception_img.copy()
        out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(15, 10))
        
        if path_to_draw:
            pts = np.array([[p[1], p[0]] for p in path_to_draw], np.int32)
            # Vibrant Green line (10px) for maximum visibility
            cv2.polylines(out, [pts], False, (0, 255, 0), 10) 
            # Thin black border for the line to make it pop on any background
            cv2.polylines(out, [pts], False, (0, 0, 0), 2) 
            
        plt.imshow(out)
            
        for t in targets:
            plt.scatter(t[1], t[0], color='red', s=400, marker='x', linewidths=3)
        
        # Plot start point as a Blue Circle
        plt.scatter(start_point[1], start_point[0], color='blue', s=500, marker='o', edgecolors='white', linewidths=2)
        
        # Plot end point as a Green Diamond
        ep = end_point if end_point else start_point
        plt.scatter(ep[1], ep[0], color='green', s=600, marker='D', edgecolors='white', linewidths=2)
            
        plt.title("Rescue Mission: Optimized Navigation Path")
        plt.axis('off')
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close() # Release memory and don't block terminal
        print(f"Final mission visualization saved to {save_path}")