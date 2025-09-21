"""CARLA Map Spawn Points Inspector Tool"""
import carla
import numpy as np
import matplotlib.pyplot as plt
import time

class MapInspector:
    def __init__(self, host='localhost', port=2000):
        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)
        self.world = None
        self.world_map = None
        
    def connect_and_load_map(self, map_name='Town04'):
        """Connect and load map"""
        try:
            self.world = self.client.get_world()
            current_map = self.world.get_map().name.split('/')[-1]
            
            if not current_map.startswith(map_name):
                print(f"Loading map {map_name}...")
                self.world = self.client.load_world(map_name)
                time.sleep(3)
            
            self.world_map = self.world.get_map()
            print(f"Connected to map: {map_name}")
            return True
            
        except Exception as e:
            print(f"Connection failed: {e}")
            return False
    
    def inspect_spawn_points(self):
        """Check all spawn points"""
        if not self.world_map:
            print("Please connect to map first")
            return
        
        spawn_points = self.world_map.get_spawn_points()
        print(f"\nMap has total {len(spawn_points)} spawn points:")
        print("=" * 80)
        
        # Analyze spawn points
        valid_points = []
        highway_points = []
        
        for i, spawn_point in enumerate(spawn_points):
            loc = spawn_point.location
            waypoint = self.world_map.get_waypoint(loc)
            
            # Get road information
            road_id = waypoint.road_id
            lane_id = waypoint.lane_id
            lane_type = waypoint.lane_type
            lane_width = waypoint.lane_width
            
            # Check if it's a valid driving lane
            is_driving = lane_type == carla.LaneType.Driving
            
            # Simple check for highway (based on lane width)
            is_highway = lane_width > 3.0 and is_driving
            
            if is_driving:
                valid_points.append(i)
            if is_highway:
                highway_points.append(i)
            
            print(f"Point {i:2d}: ({loc.x:8.1f}, {loc.y:8.1f}, {loc.z:5.1f}) "
                  f"Road{road_id:3d} Lane{lane_id:2d} Width{lane_width:.1f}m "
                  f"{'Driving' if is_driving else 'Other'} "
                  f"{'Highway' if is_highway else ''}")
        
        print("=" * 80)
        print(f"Valid driving lane spawn points: {len(valid_points)}")
        print(f"Possible highway sections: {len(highway_points)}")
        
        if highway_points:
            print(f"\nRecommended highway spawn points: {highway_points[:5]}")  # Show first 5
        elif valid_points:
            print(f"\nRecommended valid spawn points: {valid_points[:10]}")  # Show first 10
        
        return spawn_points, valid_points, highway_points
    
    def test_spawn_point(self, spawn_index):
        """Test specific spawn point"""
        if not self.world:
            print("Please connect to map first")
            return
        
        spawn_points = self.world_map.get_spawn_points()
        
        if spawn_index >= len(spawn_points):
            print(f"Error: Spawn point index {spawn_index} out of range (0-{len(spawn_points)-1})")
            return
        
        spawn_point = spawn_points[spawn_index]
        
        print(f"\nTesting spawn point {spawn_index}:")
        print(f"Position: ({spawn_point.location.x:.1f}, {spawn_point.location.y:.1f}, {spawn_point.location.z:.1f})")
        print(f"Rotation: ({spawn_point.rotation.pitch:.1f}, {spawn_point.rotation.yaw:.1f}, {spawn_point.rotation.roll:.1f})")
        
        # Get waypoint information
        waypoint = self.world_map.get_waypoint(spawn_point.location)
        print(f"Road ID: {waypoint.road_id}, Lane ID: {waypoint.lane_id}")
        print(f"Lane Type: {waypoint.lane_type}, Lane Width: {waypoint.lane_width:.1f}m")
        
        # Check forward path
        self._check_forward_path(waypoint, distance=500)
        
        # Check adjacent lanes
        self._check_adjacent_lanes(waypoint)
    
    def _check_forward_path(self, start_waypoint, distance=500):
        """Check forward path"""
        print(f"\nChecking forward {distance}m path:")
        
        current_waypoint = start_waypoint
        total_distance = 0
        waypoint_count = 0
        
        while total_distance < distance and waypoint_count < 100:
            next_waypoints = current_waypoint.next(5.0)
            if not next_waypoints:
                break
            
            current_waypoint = next_waypoints[0]
            total_distance += 5.0
            waypoint_count += 1
            
            if waypoint_count % 20 == 0:  # Show every 100m
                loc = current_waypoint.transform.location
                print(f"  {total_distance:3.0f}m: ({loc.x:8.1f}, {loc.y:8.1f}) Road{current_waypoint.road_id}")
        
        print(f"Path check complete: Total distance {total_distance:.0f}m, {waypoint_count} waypoints")
        
        if total_distance >= 400:  # At least 400m available path
            print("✓ Path length sufficient, suitable for straight road testing")
        else:
            print("⚠ Path might not be long enough, recommend choosing other spawn points")
    
    def _check_adjacent_lanes(self, waypoint):
        """Check left and right lanes"""
        print("\nChecking adjacent lanes:")
        
        # Check left lane
        left_waypoint = waypoint.get_left_lane()
        if left_waypoint and left_waypoint.lane_type == carla.LaneType.Driving:
            print(f"✓ Left lane available: Road{left_waypoint.road_id} Lane{left_waypoint.lane_id}")
        else:
            print("✗ Left lane not available")
        
        # Check right lane
        right_waypoint = waypoint.get_right_lane()
        if right_waypoint and right_waypoint.lane_type == carla.LaneType.Driving:
            print(f"✓ Right lane available: Road{right_waypoint.road_id} Lane{right_waypoint.lane_id}")
        else:
            print("✗ Right lane not available")
    
    def visualize_spawn_points(self):
        """Visualize spawn point distribution"""
        if not self.world_map:
            print("Please connect to map first")
            return
        
        spawn_points = self.world_map.get_spawn_points()
        
        # Extract coordinates
        x_coords = [sp.location.x for sp in spawn_points]
        y_coords = [sp.location.y for sp in spawn_points]
        
        # Classify points
        valid_points = []
        highway_points = []
        
        for i, spawn_point in enumerate(spawn_points):
            waypoint = self.world_map.get_waypoint(spawn_point.location)
            is_driving = waypoint.lane_type == carla.LaneType.Driving
            is_highway = waypoint.lane_width > 3.0 and is_driving
            
            if is_driving:
                valid_points.append(i)
            if is_highway:
                highway_points.append(i)
        
        # Plot
        plt.figure(figsize=(12, 8))
        
        # All points (gray)
        plt.scatter(x_coords, y_coords, c='lightgray', s=30, alpha=0.5, label='All spawn points')
        
        # Valid driving points (blue)
        if valid_points:
            valid_x = [x_coords[i] for i in valid_points]
            valid_y = [y_coords[i] for i in valid_points]
            plt.scatter(valid_x, valid_y, c='blue', s=50, alpha=0.7, label='Valid driving points')
        
        # Highway sections (red)
        if highway_points:
            highway_x = [x_coords[i] for i in highway_points]
            highway_y = [y_coords[i] for i in highway_points]
            plt.scatter(highway_x, highway_y, c='red', s=80, alpha=0.8, label='Recommended highway points')
        
        # Annotate first few recommended points
        recommend_points = highway_points[:5] if highway_points else valid_points[:5]
        for i, point_idx in enumerate(recommend_points):
            plt.annotate(f'{point_idx}', 
                        (x_coords[point_idx], y_coords[point_idx]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=10, fontweight='bold')
        
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title(f'CARLA Map Spawn Points Distribution (Total: {len(spawn_points)})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.show()
        
        return recommend_points

def main():
    print("CARLA Map Spawn Points Inspector Tool")
    print("=" * 50)
    
    inspector = MapInspector()
    
    # Connect to map
    if not inspector.connect_and_load_map('Town04'):
        return
    
    # Check all spawn points
    spawn_points, valid_points, highway_points = inspector.inspect_spawn_points()
    
    # Visualize
    try:
        recommend_points = inspector.visualize_spawn_points()
        print(f"\nRecommended spawn point indices: {recommend_points}")
    except:
        print("Cannot display visualization chart")
    
    # Interactive testing
    while True:
        try:
            user_input = input(f"\nEnter spawn point index to test (0-{len(spawn_points)-1}, 'q' to quit): ")
            if user_input.lower() == 'q':
                break
            
            spawn_index = int(user_input)
            inspector.test_spawn_point(spawn_index)
            
        except ValueError:
            print("Please enter a valid number")
        except KeyboardInterrupt:
            break
    
    print("Inspector tool finished")

if __name__ == "__main__":
    main()